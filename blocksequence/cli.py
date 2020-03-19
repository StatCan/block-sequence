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


@click.command(help='Generate a sequence that enumerates all edges and orders the blocks given.')
@click_log.simple_verbosity_option(logger)
@click.argument('edges', envvar='SEQ_EDGE_DATA_FILE', 
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('blocks', envvar='SEQ_BLOCK_DATA_FILE', 
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('outfile', envvar='SEQ_OUTPUT', default='sequence_results.csv',
    type=click.Path(dir_okay=False, exists=False, file_okay=True))
@click.option('--parent_geography_uid_field', envvar='SEQ_PARENT_UID', default='parent_block_uid')
@click.option('--child_geography_uid_field', envvar='SEQ_CHILD_UID', default='child_block_uid')
@click.option('--bo_field', envvar='SEQ_BO_FIELD_NAME', default='block_order')
@click.option('--eo_field', envvar='SEQ_EO_FIELD_NAME', default='edge_order')
@click.option('--block_max_evolutions', default=1000, type=click.INT)
@click.option('--block_x_field', envvar='SEQ_BLOCK_X_FIELD', default='rep_point_x')
@click.option('--block_y_field', envvar='SEQ_BLOCK_Y_FIELD', default='rep_point_y')
@click.option('--edge_source_field', envvar='SEQ_EDGE_SOURCE_FIELD', default='source')
@click.option('--edge_target_field', envvar='SEQ_EDGE_TARGET_FIELD', default='target')
@click.option('--edge_uid_field', envvar='SEQ_EDGE_UID_FIELD', default='edge_uid')
@click.option('--edge_interior_flag_field', envvar='SEQ_INTERIOR_EDGE_FLAG_FIELD', default='interior_flag')
@click.option('--edge_struid_field', envvar='SEQ_STREET_UID_FIELD', default='street_uid')
@click.option('--anomaly_folder', envvar='SEQ_ANOMALY_FOLDER', default='.', 
    type=click.Path(dir_okay=True, file_okay=False),
    help='Folder to write data on any blocks that do not get processed ideally.')
def main(edges, blocks, outfile, parent_geography_uid_field, child_geography_uid_field, bo_field, 
    eo_field, block_x_field, block_y_field, block_max_evolutions,
    edge_source_field, edge_target_field, edge_uid_field, edge_interior_flag_field, edge_struid_field, 
    anomaly_folder):
    """Sequence a road network through a geography to aid in enumeration."""

    logger.debug("cli.main start")

    # The number of evolutions to use when determining the optimal block order. The higher the number the more reliable
    # the output will be, but it will take longer to calculate.
    block_order_evolution_count = block_max_evolutions

    logger.debug("Reading block list file to get all blocks that will be processed.")
    block_geographies = pd.read_csv(blocks)

    # Some edges are marked as anomalies (see EdgeSequence class). This is where to store the graph data for those
    # edge sets.
    anomaly_folder = Path(anomaly_folder).joinpath('anomalies')
    anomaly_folder.mkdir(parents=True, exist_ok=True)

    for pgeo_id, child_blocks in block_geographies.groupby(parent_geography_uid_field):
        block_sequencer = BlockOrder(child_blocks, child_geography_uid_field, block_x_field, block_y_field,
                                    block_order_field_name=bo_field,
                                    max_evolution_count=block_order_evolution_count)
        bo_df = block_sequencer.get_optimal_order()

        # load the edge data for this parent geography
        pgeo_edges = load_edges_for_parent_geo(pgeo_id, parent_geography_uid_field, edges)

        # For some reason the node numbers become floats when reading them back from HDF, so force integer type
        # pd.to_numeric is not used because the node numbers need to be a consistent type when writing to the HDF file.
        pgeo_edges[edge_source_field] = pgeo_edges[edge_source_field].astype(np.int32)
        pgeo_edges[edge_target_field] = pgeo_edges[edge_target_field].astype(np.int32)

        logger.debug("Merging block order with edge sequence results")
        # Join the block order results onto the edge sequence (n:1 join).
        pgeo_edges = pgeo_edges.merge(bo_df, how='left', on=child_geography_uid_field)

        # Calculate the edge sequence within each child geography for this parent geography
        logger.debug("Calculating edge order for all blocks in %s", pgeo_id)
        sequencer = EdgeOrder(pgeo_edges, source=edge_source_field, target=edge_target_field, 
            anomalies_path=anomaly_folder, eo_field_name=eo_field, interior_edge_flag=edge_interior_flag_field,
            block_order_field=bo_field, edge_uid_field=edge_uid_field, struid_field=edge_struid_field)
        pgeo_edges = sequencer.sequence(child_geography_uid_field)

        # Don't keep the interim columns that were just used as flags.
        flag_attrs = [edge_struid_field, edge_interior_flag_field]
        result_columns = [c for c in pgeo_edges.columns if c not in flag_attrs]
        pgeo_edges = pgeo_edges.filter(items=result_columns)

        ##
        # perform some checks on the data to make sure things went ok
        ##

        # check for null values
        if pgeo_edges[eo_field].isnull().any():
            logging.error("%s of %s edges have no edge_order assigned in parent geography %s",
                          len(pgeo_edges[eo_field].isna()), len(pgeo_edges), pgeo_id)

        # Save the data in temp storage for later
        logging.debug("Writing sequence results to output file")
        str_col_min_length = {c: pgeo_edges[c].str.len().max() for c in
                              pgeo_edges.columns[pgeo_edges.dtypes == 'object']}
        try:
            # pgeo_edges.to_hdf(outfile, key='sequence', format='table', append=True,
            #                   data_columns=[parent_geo_uid_field, child_geo_uid_field], min_itemsize={'path_type': 9})
            pgeo_edges.to_csv(outfile, index=False, mode='a')
        except (KeyError, ValueError):
            # don't fail everything just because a single geography failed
            logger.error("Columns on data that failed:\n%s", pgeo_edges.dtypes)
            logger.exception("Failed processing %s. Data will be missing from output.", pgeo_id)

    logger.debug("cli.main end")


def load_edges_for_parent_geo(pgeo_uid, pgeo_uid_field, edges_csv_path):
    """Load the edge data for a given parent geography from the CSV file."""

    logger.debug("Fetching data from %s where %s = %s", edges_csv_path, pgeo_uid_field, pgeo_uid)
    df = pd.read_csv(edges_csv_path)
    df = df[df[pgeo_uid_field] == pgeo_uid]
    return df

def start():
    main()


if __name__ == '__main__':
    # start the application
    start()
