import logging

import click
import click_log
from dotenv import load_dotenv
import geopandas as gpd
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from .utils import commands as utils
from .sequence import commands as sequence

logger = logging.getLogger()
click_log.basic_config(logger)

# support loading of environment from .env files
load_dotenv()

@click.group()
@click_log.simple_verbosity_option(logger)
@click.option('--source_host', envvar='SEQ_SOURCE_HOST', help="Source host name")
@click.option('--source_db', envvar='SEQ_SOURCE_DB', help="Source database name")
@click.option('--source_user', envvar='SEQ_SOURCE_USER', help="Source DB username")
@click.option('--source_pass', envvar='SEQ_SOURCE_PASS', help="Source DB password")
@click.option('--outdb', envvar='SEQ_OUTPUT', default='sequencing.sqlite', 
  type=click.Path(dir_okay=False, resolve_path=True),
  help="Output SQLite database filename")
@click.pass_context
def main(ctx, source_host, source_db, source_user, source_pass, outdb):
  """Sequence a road network through a geography to aid in enumeration."""

  logger.debug("cli.main start")

  # ensure that ctx.obj exists and is a dict
  ctx.ensure_object(dict)

  # initiate the source and output database connections and store those in the context
  logger.debug("Source DB: {} as {} on {}".format(source_db, source_user, source_host))
  src_conn = create_engine('postgresql://{user}:{password}@{host}/{db}'.format(db=source_db, user=source_user, password=source_pass, host=source_host))
  logger.debug("Output DB: {}".format(outdb))
  dest_conn = create_engine('sqlite:///{}'.format(outdb), echo=False)

  # configure logging on the database engine to match our logger
  # sqlalchemy returns full results with debug, so avoid going to that level
  log_level = logger.getEffectiveLevel() + 10
  if log_level == logging.CRITICAL:
    log_level -=  10
  logging.getLogger('sqlalchemy.engine').setLevel(log_level)

  # attach DB information to the context
  ctx.obj = {
    'src_db': src_conn,
    'dest_db': dest_conn
  }

  logger.debug("cli.main end")

# add subcommands
main.add_command(utils.node_weights)
main.add_command(sequence.sequence)
main.add_command(utils.order_blocks)
main.add_command(utils.t_intersections)
main.add_command(utils.start_points)


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
  edge_df = pd.read_sql_table(edge_layer, sqlite_conn, coerce_float=False).rename(columns=str.lower)
  line_df= gpd.read_file(gpkg, layer=line_layer).rename(columns=str.lower)
  # the block layer is only used for attributes, so drop the geometry
  block_df = (gpd.read_file(gpkg, layer=block_layer).drop('geometry')).rename(columns=str.lower)

  # merge all the dataframes together, creating a single dataframe to work from
  edge_gdf = (line_df.merge(edge_df, on='ngd_uid', sort=False)
              .merge(block_df, on=child_geography_uid_field, sort=False, copy=False))

  # clean up some memory
  del edge_df, line_df, block_df

  # use the line geometry to extract start and end x/y coordinates
  edge_gdf[['start_node_x', 'start_node_y']] = np.vectorize(get_start_node)(edge_gdf['geometry'])
  edge_gdf[['end_node_x', 'end_node_y']] = np.vectorize(get_end_node)(edge_gdf['geometry'])

  # generate IDs to use for the coordinates, instead of the coordinates themselves
  start_nodes = edge_gdf[['start_node_x', 'start_node_y']].apply(lambda x: (x[0], x[1]))
  end_nodes = edge_gdf[['end_node_x', 'end_node_y']].apply(lambda x: (x[0], x[1]))
  coord_set = frozenset(start_nodes).union(end_nodes)
  coord_lookup = dict(zip(coord_set, range(1, len(coord_set) + 1)))
  del coord_set, start_nodes, end_nodes

  # assign the IDs
  edge_gdf['start_node_id'] = edge_gdf['start_node'].map(coord_lookup)
  edge_gdf['end_node_id'] = edge_gdf['end_node'].map(coord_lookup)

  # convert our GeoDataFrame to a normal dataframe to be written out
  edges = pd.DataFrame(edge_gdf.drop('geometry', axis=1))
  # write the data, overwriting anything that already exists
  edges.to_sql('edge_list', ctx['dest_conn'], if_exists='replace')


def get_start_node(line):
  coord = line.coords[0]
  return (coord[0], coord[1])

def get_end_node(line):
  coord = line.coords[-1]
  return (coord[0], coord[1])


def start():
  main(obj={})

if __name__ == '__main__':
  # start the application
  start()
