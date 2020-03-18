import logging

import click
import click_log
from dotenv import load_dotenv
import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from pathlib import Path
from shapely.ops import linemerge
import sqlalchemy as sa

from .algorithms import EdgeOrder, BlockOrder

logger = logging.getLogger()
click_log.basic_config(logger)

# support loading of environment from .env files
load_dotenv()


@click.group()
@click_log.simple_verbosity_option(logger)
@click.option('--outdb', envvar='SEQ_OUTPUT', default='sequencing.sqlite',
              type=click.Path(dir_okay=False, resolve_path=True),
              help="Output SQLite database filename")
@click.pass_context
def main(ctx, outdb):
    """Sequence a road network through a geography to aid in enumeration."""

    logger.debug("cli.main start")

    # ensure that ctx.obj exists and is a dict
    ctx.ensure_object(dict)

    # initiate the output database connections and store those in the context
    logger.debug("Output DB: {}".format(outdb))
    dest_conn = sa.create_engine('sqlite:///{}'.format(outdb), echo=False)

    # configure logging on the database engine to match our logger
    # sqlalchemy returns full results with debug, so avoid going to that level
    log_level = logger.getEffectiveLevel() + 10
    if log_level == logging.CRITICAL:
        log_level -= 10
    logging.getLogger('sqlalchemy.engine').setLevel(log_level)

    # attach DB information to the context
    ctx.obj = {
        'dest_db': dest_conn
    }

    logger.debug("cli.main end")


@click.command()
@click.argument('gpkg', envvar='SEQ_GPKG_PATH')
@click.argument('parent_geography_uid_field', envvar='SEQ_PARENT_UID')
@click.argument('child_geography_uid_field', envvar='SEQ_CHILD_UID')
@click.option('--edge_layer', default='BF', show_default=True)
@click.option('--block_layer', default='LB', show_default=True)
@click.option('--line_layer', default='NGD_AL', show_default=True)
@click.pass_context
def create_edge_list(ctx, gpkg, parent_geography_uid_field, child_geography_uid_field, edge_layer, block_layer,
                     line_layer):
    """
    Create a listing of edges to be sequenced through the sequence command from the provided input tables.

    This is easier to do in a PostGIS view, but since we're working with a GeoPackage the edge listing is extracted
    to the output database. Ultimately, we need to create a list of edges that can be turned into a graph and are
    connected through source and target nodes. The start and end points of the line are used as our source and target
    nodes, in this case.
    """

    logger.debug("create_edge_list start")

    # sqlite path to the gpkg for pandas
    sqlite_conn = f'sqlite:///{gpkg}'

    # In order to ensure that data is given to NetworkX with right hand arcs first, define an ordered category that
    # can be used for sorting later on.
    arc_category = CategoricalDtype(categories=['R', 'L'], ordered=True)

    logger.debug("Reading source data from GPKG")
    # get each of the required layers as a dataframe
    edge_df = (pd.read_sql_table(edge_layer, sqlite_conn, coerce_float=False)
               .rename(columns=str.lower))
    line_df = (gpd.read_file(gpkg, layer=line_layer)
               .rename(columns=str.lower))
    # the block layer is only used for attributes, so drop the geometry
    block_df = (gpd.read_file(gpkg, layer=block_layer)
                .drop('geometry', axis=1)
                .rename(columns=str.lower))

    logger.debug("Merging source layers into single DataFrame")
    # Merge all the dataframes together, creating a single dataframe to work from. This will also store the line
    # length as an attribute for use in later analysis and create unique node identifiers.
    edge_gdf = (line_df.merge(edge_df, on='ngd_uid', sort=False)
                .merge(block_df, on=child_geography_uid_field.lower(), sort=False, copy=False)
                .assign(length=lambda x: x.length)
                .pipe(generate_node_numbers))
    edge_gdf['arc_side'] = edge_gdf['arc_side'].astype(arc_category)

    # Generate the start and end node numbers used to link all the geometries together within the NetworkX graph.

    # clean up some memory
    del edge_df, line_df, block_df

    # Calculate which road arcs are on the interior of a block to help avoid them becoming start points later.
    logger.debug("Calculating road arc properties")
    edge_gdf['interior'] = np.where(edge_gdf['bb_uid_l'] == edge_gdf['bb_uid_r'], 1, 0)
    # Calculate the start and end coordinates for downstream processes to be able to create map output.
    logger.debug("Extract start and end coordinates from each arc")
    edge_gdf['start_x_coord'] = edge_gdf.geometry.apply(lambda geo: linemerge(geo).coords[0][0])
    edge_gdf['start_y_coord'] = edge_gdf.geometry.apply(lambda geo: linemerge(geo).coords[0][-1])
    edge_gdf['end_x_coord'] = edge_gdf.geometry.apply(lambda geo: linemerge(geo).coords[-1][0])
    edge_gdf['end_y_coord'] = edge_gdf.geometry.apply(lambda geo: linemerge(geo).coords[-1][-1])

    logger.debug("Saving output to destination database")
    # convert our GeoDataFrame to a normal dataframe to be written out
    edges = pd.DataFrame(edge_gdf.drop('geometry', axis=1))

    edges.to_hdf('sequence_input.h5', 'edges', format='table',
                 data_columns=[parent_geography_uid_field, child_geography_uid_field])

    ##
    # Build the block represenatative point information used for block ordering.
    ##
    block_df = (gpd.read_file(gpkg, layer=block_layer)
                .rename(columns=str.lower))

    # Calculate the representative points for each block.
    # Rep points are cheaper to calculate than centroids, and a guaranteed to be within the block exterior, which is
    # why they are used here.
    block_df['rep_point'] = block_df.geometry.representative_point()
    block_df['rep_point_x'] = block_df['rep_point'].apply(lambda geo: geo.x)
    block_df['rep_point_y'] = block_df['rep_point'].apply(lambda geo: geo.y)
    block_df = pd.DataFrame(
        block_df.filter(['rep_point_x', 'rep_point_y', parent_geography_uid_field, child_geography_uid_field])
        .rename(columns=str.lower))

    # Send the block information to an HDF file for fast lookups later.
    logger.debug("Writing block data to HDF file")
    block_df.to_hdf('block_points.h5', 'blocks', format='table',
                    data_columns=[parent_geography_uid_field, child_geography_uid_field])

    logger.debug("create_edge_list end")


def generate_node_numbers(df):
    """Create unique IDs for all the nodes from both the start and end of the road arcs.

    Parameters
    ----------
    df : GeoPandas GeoDataFrame
        A GeoDataFrame representing line segments that form the road network

    Returns
    -------
    df : GeoPandas GeoDataFrame
        The same GeoDataFrame as was passed in as input, but with new 'startnodenum' and 'endnodenum' columns
    """

    # Start and end node tuples are required temporarily to be able to calculate the node numbers.
    logger.debug("Getting start and end nodes for each arc")
    df['start_node'] = df.geometry.apply(lambda geo: linemerge(geo).coords[0])
    df['end_node'] = df.geometry.apply(lambda geo: linemerge(geo).coords[-1])

    # Create a set from all the nodes so that only unique start and end node tuples are represented
    logger.debug("Generating node IDs for all start and end coordinates")
    coord_set = frozenset(df['start_node']).union(df['end_node'])
    # Assign each coordinate tuple a unique identifier and store it in a dictionary for lookup
    coord_lookup = dict(zip(coord_set, range(1, len(coord_set) + 1)))

    # Map the coordinate IDs back onto the coorindate tuples in the source DataFrames
    logger.debug("Mapping node IDs to start and end nodes")
    df['startnodenum'] = df['start_node'].map(coord_lookup)
    df['endnodenum'] = df['end_node'].map(coord_lookup)

    # remove the temporary start and end columns from the data
    logger.debug("Removing %s columns from the dataframe", ['start_node', 'end_node'])
    df = df.drop(['start_node', 'end_node'], axis=1)

    return df


@click.command()
@click.argument('parent_geography_uid_field', envvar='SEQ_PARENT_UID')
@click.argument('child_geography_uid_field', envvar='SEQ_CHILD_UID')
@click.pass_context
def sequence(ctx, parent_geography_uid_field, child_geography_uid_field):
    logger.debug("sequence start")

    # The number of evolutions to use when determining the optimal block order. The higher the number the more reliable
    # the output will be, but it will take longer to calculate.
    block_order_evolution_count = 1000

    logger.debug("Reading block list file to get all blocks that will be processed.")
    block_geographies = load_from_hdf(Path('block_points.h5'), 'blocks')

    # Some edges are marked as anomalies (see EdgeSequence class). This is where to store the graph data for those
    # edge sets.
    anomaly_folder = Path('.').joinpath("anomalies")
    anomaly_folder.mkdir(parents=True, exist_ok=True)

    for pgeo_id, child_blocks in block_geographies.groupby(parent_geography_uid_field):
        block_sequencer = BlockOrder(child_blocks, child_geography_uid_field,
                                     max_evolution_count=block_order_evolution_count)
        bo_df = block_sequencer.get_optimal_order()

        # load the edge data for this parent geography
        hdf_filter = f"{parent_geography_uid_field}=={pgeo_id}"
        pgeo_edges = load_from_hdf(Path('sequence_input.h5'), 'edges', hdf_filter)

        # For some reason the node numbers become floats when reading them back from HDF, so force integer type
        # pd.to_numeric is not used because the node numbers need to be a consistent type when writing to the HDF file.
        pgeo_edges['startnodenum'] = pgeo_edges['startnodenum'].astype(np.int32)
        pgeo_edges['endnodenum'] = pgeo_edges['endnodenum'].astype(np.int32)

        logger.debug("Merging block order with edge sequence results")
        # Join the block order results onto the edge sequence (n:1 join).
        pgeo_edges = pgeo_edges.merge(bo_df, how='left', on=child_geography_uid_field)

        # Calculate the edge sequence within each child geography for this parent geography
        logger.debug("Calculating edge order for all blocks in %s", pgeo_id)
        sequencer = EdgeOrder(pgeo_edges, source='startnodenum', target='endnodenum', anomalies_path=anomaly_folder)
        pgeo_edges = sequencer.sequence(child_geography_uid_field)

        # Don't keep the interim columns that were just used as flags.
        flag_attrs = ['ngd_str_uid', 'interior']
        result_columns = [c for c in pgeo_edges.columns if c not in flag_attrs]
        pgeo_edges = pgeo_edges.filter(items=result_columns)

        ##
        # perform some checks on the data to make sure things went ok
        ##

        # check for null values
        if pgeo_edges['edge_order'].isnull().any():
            logging.error("%s of %s edges have no edge_order assigned in parent geography %s",
                          len(pgeo_edges['edge_order'].isna()), len(pgeo_edges), pgeo_id)

        # Save the data in temp storage for later
        logging.debug("Writing sequence results to HDF file")
        str_col_min_length = {c: pgeo_edges[c].str.len().max() for c in
                              pgeo_edges.columns[pgeo_edges.dtypes == 'object']}
        try:
            pgeo_edges.to_hdf('sequence_results.h5', key='sequence', format='table', append=True,
                              # data_columns=[parent_geo_uid_field, child_geo_uid_field], min_itemsize={'path_type': 9})
                              min_itemsize={'path_type': 9})
        except (KeyError, ValueError):
            # don't fail everything just because a single geography failed
            logger.error("Columns on data that failed:\n%s", pgeo_edges.dtypes)
            logger.exception("Failed processing %s. Data will be missing from output.", pgeo_id)

    logger.debug("sequence_blocks end")


def load_from_hdf(hdf_file, table_name, filter=''):
    """Load some data from an HDF file, returning a DataFrame.

    Parameters
    ----------
    hdf_file : pathlib.Path
        The path to the HDF file to read from.

    table_name : String
        The name of the table within the HDF file to read from.

    filter : String
        Filter used to limit the data that is read from the HDF file. This follows pandas query semantics.

    Returns
    -------
    df : Pandas DataFrame
        The contents of the HDF that were pulled out of the file.
    """

    logger.debug("load_from_hdf started")

    logger.debug("Reading table %s from %s where %s", table_name, hdf_file, filter)
    df = pd.read_hdf(hdf_file, table_name, where=filter)

    return df


# register the commands
main.add_command(create_edge_list)
main.add_command(sequence)


def start():
    main(obj={})


if __name__ == '__main__':
    # start the application
    start()
