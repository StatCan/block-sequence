import logging
import os
from pathlib import Path

import click
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.sql import select
from geoalchemy2 import Geometry

logger = logging.getLogger(__name__)

@click.command()
@click.argument('parent_layer', envvar='SEQ_PARENT_LAYER')
@click.argument('parent_uid', envvar='SEQ_PARENT_UID')
@click.pass_context
def node_weights(ctx, parent_layer, parent_uid):
  """Assess the weight of each node in an parent geography.
  
  Parent geography boundaries are used to calculate potential start points for routing through
  the road network. The more popular a given point, the less an enumerator would need to travel between 
  geographies, making it more desirable to use.
  """

  logger.debug("node_weights start")

  # sqlalchemy setup
  meta = MetaData()

  # get every parent geography in the dataset
  logger.debug("Getting %s layer data", parent_layer)
  pgeo_table = Table(parent_layer, meta, autoload=True, autoload_with=ctx.obj['src_db'])
  pgeo_select = select([pgeo_table.c[parent_uid], pgeo_table.c.geom])
  pgeo = gpd.GeoDataFrame.from_postgis(pgeo_select, ctx.obj['src_db'])

  # pull out the nodes in the polygons
  logger.debug("Calculating node coordinates for every polygon")
  pgeo['coords'] = pgeo.geometry.boundary.apply(lambda x: x[0].coords)

  # limit to the number of columns we really need
  pgeo = pgeo.filter([parent_uid, 'coords'])

  # generate node IDs from coordinate pairs
  pgeo['node'] = pgeo['coord'].apply(lambda c: get_coord_node(c))

  logger.debug("Grouping nodes to determine popularity")
  pgeo['weight'] = pgeo.groupby(['node'])[parent_uid].transform('count')

  # cast the node to str for writing to the db
  pgeo['node'] = pgeo['node'].astype(str)
  pgeo = pgeo.drop('coord', axis=1)
  
  # write it all to sqlite for reference by later steps
  logger.debug("Saving to node_weights table")
  pgeo.to_sql('node_weights', con=ctx.obj['dest_db'], if_exists='replace', index=False)

  logger.debug("node_weights end")

def get_coord_node(coord):
  """Create a node identifier from a coordinate pair."""
  x = coord[0]
  y = coord[1]
  c = (round(x, 4), round(y, 4))
  return c

def get_circuit_distance(circuit, length_field):
  """Compute the total distance for a complete eulerian circuit."""
  
  logger.debug('calculating circuit distance')
  return sum([edge[2][0][length_field] for edge in circuit])

def get_graph_distance(g, length_field):
  """Compute the total distance for a given graph."""

  logger.debug('calculating graph distance')
  return sum(nx.get_edge_attributes(g, length_field).values())

def get_edge_count(g):
  """Calculate the total number of edges in a graph."""

  logger.debug('getting edge count for graph')
  return len(g.edges())

def get_node_count(g):
  """Calculate the total number of nodes in a graph."""

  logger.debug('getting node count for graph')
  return len(g.nodes())

@click.command()
@click.argument('pid', envvar='SEQ_PARENT_UID')
@click.argument('cid', envvar='SEQ_CHILD_UID')
@click.pass_context
def order_blocks(ctx, pid, cid):
  """Calculate the block ordering based on the edge sequence."""

  logger.debug('order_blocks started')

  # sqlalchemy setup
  meta = MetaData()

  # pull the edge sequence out of the database
  logger.debug("Reading from edge_sequence table")
  edge_table = Table('edge_sequence', meta, autoload=True, autoload_with=ctx.obj['dest_db'])
  edge_select = select([edge_table])
  edge_sequence = pd.read_sql(edge_select, con=ctx.obj['dest_db'])

  # calculate the block order
  logger.debug("Calculating the block order")
  edge_sequence = edge_sequence.groupby(pid, sort=True).apply(sub_block_order, {'key':cid})

  # calculate the edge order within the blocks
  logger.debug("Calculating edge order")
  edge_sequence['edge_order'] = edge_sequence.groupby([pid, cid], sort=True).cumcount()+1
  
  # calculate the chain ID
  logger.debug("Calculating chain ID field")
  edge_sequence['chain_id'] = np.where(edge_sequence['edge_order'] == 1, 1, 0)

  output_table_name = 'ordered_sequence'
  logger.debug("Saving block order to %s table", output_table_name)
  edge_sequence.to_sql(output_table_name, con=ctx.obj['dest_db'])

  logger.debug('order_blocks ended')

def sub_block_order(data, key='lb_uid'):
  """Takes a DataFrame grouped by a parent geography and creates an ID number for the order in which they appear."""

  grp = data.groupby(key)
  data['block_order'] = grp.ngroup()+1
  return data

@click.command()
@click.argument('pgeo', envvar='SEQ_PARENT_LAYER')
@click.pass_context
def start_points(ctx, pgeo):
  """Generate a table of all the start points in the sequence."""
  
  logger.debug('start_points begin')

  # sqlalchemy setup
  meta = MetaData()

  os_table = Table('ordered_sequence', meta, autoload=True, autoload_with=ctx.obj['dest_db'])
  os_table_query = select([os_table], os_table.c.edge_order == 1)
  sp = pd.read_sql(os_table_query, ctx.obj['dest_db'])
  
  # seems like a waste of time - maybe rename the columns?
  sp['x'] = sp['source_x']
  sp['y'] = sp['source_y']

  # add a t_flag field
  sp['t_flag'] = None

  # set the LUID
  pgeo_table = Table(pgeo, meta, autoload=True, autoload_with=ctx.obj['src_db'])
  pgeo_table_query = select([pgeo_table.c.luid, pgeo_table.c.lu_uid])
  lu_info = pd.read_sql(pgeo_table_query, con=ctx.obj['src_db'])

  # join onto the start points
  sp = sp.merge(lu_info, on='lu_uid', how='left')

  sp.to_sql('start_points', con=ctx.obj['dest_db'])

  logger.debug('start_points end')

@click.command()
@click.argument('pgeo', envvar='SEQ_CHILD_LAYER')
@click.argument('roads', envvar='SEQ_ROAD_LAYER')
@click.pass_context
def t_intersections(ctx, pgeo, roads):
  """Find places where the road network forms a T intersection with the parent geography boundary."""

  logger.debug("t_intersections start")

  # hold onto sqlalchemy metadata
  meta = MetaData()

  logger.debug("Loading %s table from DB", pgeo)
  poly_tbl = Table(pgeo, meta, autoload=True, autoload_with=ctx.obj['src_db'])
  poly_tbl_query = select([poly_tbl.c.lb_uid, poly_tbl.c.geom])
  poly_df = gpd.GeoDataFrame.from_postgis(poly_tbl_query, con=ctx.obj['src_db'])

  logger.debug("Loading %s table from DB", pgeo)
  roads_tbl = Table(roads, meta, autoload=True, autoload_with=ctx.obj['src_db'])
  roads_tbl_query = select([roads_tbl.c.ngd_uid, roads_tbl.c.geom], roads_tbl.c.sgmnt_typ_cde == 2)
  roads_df = gpd.GeoDataFrame.from_postgis(roads_tbl_query, con=ctx.obj['src_db'])

  # needs to test the boundary of the polygon, not the polygon itself
  poly_edges = list(poly_df.geometry.boundary)
  logger.debug("Finding T intersections for %s boundaries", len(poly_edges))
  roads_df['is_t'] = roads_df.geometry.apply(lambda r: forms_t(r, poly_edges))

  tbl_name = 't_intersection'
  logger.debug("Saving to %s table", tbl_name)
  roads_df[roads_df['is_t'] == True].to_sql(tbl_name, con=ctx.obj['dest_db'])

  logger.debug('t_intersections end')

def forms_t(arc, edges):
  """Check if arc forms a T intersection with the provided edge list."""

  # logger.debug("forms_t start")

  is_t = False
  for edge in edges:
    if arc.touches(edge):
      is_t = True
      # bail on match, no point in finding more
      break
  # logger.debug("T intersection found: %s", is_t)
  
  # logger.debug("forms_t end")
  return is_t