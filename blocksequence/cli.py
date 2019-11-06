import logging

import click
import click_log
from dotenv import load_dotenv
import geopandas as gpd
import numpy as np
import pandas as pd
import sqlalchemy as sa

from .blocksequence import BlockSequence

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
    log_level -=  10
  logging.getLogger('sqlalchemy.engine').setLevel(log_level)

  # attach DB information to the context
  ctx.obj = {
    'dest_db': dest_conn
  }

  logger.debug("cli.main end")


@click.command()
@click.argument('gpkg', envvar='SEQ_GPKG_PATH')
@click.argument('parent_geography', envvar='SEQ_PARENT_LAYER')
@click.argument('parent_geography_uid_field', envvar='SEQ_PARENT_UID')
@click.argument('child_geography_uid_field', envvar='SEQ_CHILD_UID')
@click.option('--edge_layer', default='BF', show_default=True)
@click.option('--block_layer', default='LB', show_default=True)
@click.option('--line_layer', default='NGD_AL', show_default=True)
@click.pass_context
def create_edge_list(ctx, gpkg, parent_geography, parent_geography_uid_field, child_geography_uid_field, edge_layer, block_layer, line_layer):
  """
  Create a listing of edges to be sequenced through the sequence command from the provided input tables.

  This is easier to do in a PostGIS view, but since we're working with a GeoPackage the edge listing is extracted
  to the output database. Ultimately, we need to create a list of edges that can be turned into a graph and are
  connected through source and target nodes. The start and end points of the line are used as our source and target
  nodes, in this case.
  """

  # sqlite path to the gpkg for pandas
  sqlite_conn = f'sqlite:///{gpkg}'

  # get each of the required layers as a dataframe
  edge_df = (pd.read_sql_table(edge_layer, sqlite_conn, coerce_float=False)
             .rename(columns=str.lower))
  line_df= (gpd.read_file(gpkg, layer=line_layer)
            .rename(columns=str.lower))
  # the block layer is only used for attributes, so drop the geometry
  block_df = (gpd.read_file(gpkg, layer=block_layer)
              .drop('geometry', axis=1)
              .rename(columns=str.lower))

  # merge all the dataframes together, creating a single dataframe to work from
  edge_gdf = (line_df.merge(edge_df, on='ngd_uid', sort=False)
              .merge(block_df, on=child_geography_uid_field.lower(), sort=False, copy=False))

  # store the length of the lines for use as the weight field
  edge_gdf = edge_gdf.assign(length=lambda x: x.length)

  # clean up some memory
  del edge_df, line_df, block_df

  # use the line geometry to extract start and end x/y coordinates
  edge_gdf['start_node_x'], edge_gdf['start_node_y'] = np.vectorize(get_start_node)(edge_gdf['geometry'])
  edge_gdf['end_node_x'], edge_gdf['end_node_y'] = np.vectorize(get_end_node)(edge_gdf['geometry'])

  # generate IDs to use for the coordinates, instead of the coordinates themselves
  edge_gdf['start_node'] = list(zip(edge_gdf['start_node_x'], edge_gdf['start_node_y']))
  edge_gdf['end_node'] = list(zip(edge_gdf['end_node_x'], edge_gdf['end_node_y']))
  coord_set = frozenset(edge_gdf['start_node']).union(edge_gdf['end_node'])
  coord_lookup = dict(zip(coord_set, range(1, len(coord_set) + 1)))
  del coord_set

  # assign the IDs
  edge_gdf['start_node_id'] = edge_gdf['start_node'].map(coord_lookup)
  edge_gdf['end_node_id'] = edge_gdf['end_node'].map(coord_lookup)
  edge_gdf = edge_gdf.drop(['start_node', 'end_node'], axis=1)

  # convert our GeoDataFrame to a normal dataframe to be written out
  edges = pd.DataFrame(edge_gdf.drop('geometry', axis=1))
  # write the data, overwriting anything that already exists
  edges.to_sql('edge_list', ctx.obj['dest_db'], if_exists='replace')


def get_start_node(line):
  coord = line.geoms[0].coords[0]
  return (coord[0], coord[1])

def get_end_node(line):
  coord = line.geoms[0].coords[-1]
  return (coord[0], coord[1])


@click.command()
@click.argument('parent_geography', envvar='SEQ_PARENT_LAYER')
@click.argument('parent_geography_uid_field', envvar='SEQ_PARENT_UID')
@click.argument('child_geography_uid_field', envvar='SEQ_CHILD_UID')
@click.pass_context
def sequence_blocks(ctx, parent_geography, parent_geography_uid_field, child_geography_uid_field):

    # load the edge list from the database
    edges = pd.read_sql_table('edge_list', ctx.obj['dest_db'])

    # group the edges by the parent geography
    pgeo_group = edges.groupby(by=parent_geography_uid_field.lower(), sort=False)
    # iterate each geography, calculating a eulerian circuit and writing it to the database
    for group_id, group in pgeo_group:
        bs = BlockSequence(group, 'start_node_id', 'end_node_id')
        bs_df = bs.eulerian_circuit(child_geography_uid_field.lower())
        bs_df.to_sql('sequence', ctx.obj['dest_db'], if_exists='append')

# register the commands
main.add_command(create_edge_list)
main.add_command(sequence_blocks)

def start():
  main(obj={})

if __name__ == '__main__':
  # start the application
  start()
