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
  sp = pgeo[[parent_uid, 'coords']]
  d = []
  for r in sp.iterrows():
    k = r[1][0]
    v = r[1][1]
    # logger.debug("%s has %s", k, v)
    for i in v:
      x, y = i
      c = (round(x, 4), round(y, 4))
      # logger.debug("Adding node %s to %s", c, k)
      d.append((k,c))
  coord_pop = pd.DataFrame(d, columns=[parent_uid, 'node'])
  logger.debug("Grouping nodes to determine popularity")
  coord_pop['weight'] = coord_pop.groupby(['node'])[parent_uid].transform('count')
  # cast the node to str for writing to the db
  coord_pop['node'] = coord_pop['node'].astype(str)
  
  # write it all to sqlite for reference by later steps
  logger.debug("Saving to node_weights table")
  coord_pop.to_sql('node_weights', con=ctx.obj['dest_db'], if_exists='replace', index=False)

  logger.debug("node_weights end")

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
@click.argument('cid', envvar='SEQ_CHILD_UID')
@click.pass_context
def order_blocks(ctx, cid):
  """Calculate the block ordering based on the edge sequence."""

  logger.debug('order_blocks started')

  # sqlalchemy setup
  meta = MetaData()

  # pull the edge sequence out of the database
  logger.debug("Reading from edge_sequence table")
  edge_table = Table('edge_sequence', meta, autoload=True, autoload_with=ctx.obj['src_db'])
  edge_select = select([edge_table])
  edge_sequence = pd.read_sql(edge_select, con=ctx.obj['src_db'])

  # group the blocks by the child geo ID
  grouped = edge_sequence.groupby(cid, sort=False)
  block_order = 1
  for name, group in grouped:
    edge_sequence.loc[edge_sequence[cid] == name, 'block_order'] = block_order
    edge_sequence.loc[edge_sequence[cid] == name, 'edge_order'] = range(1, len(group)+1)
    block_order += 1
  
  # calculate the chain ID
  edge_sequence['chain_id'] = np.where(edge_sequence['edge_order'] == 1, 1, 0)

  edge_sequence.to_sql('ordered_sequence', con=ctx.obj['dest_db'])

  logger.debug('order_blocks ended')


@click.command()
@click.argument('pgeo', envvar='SEQ_PARENT_LAYER')
@click.pass_context
def start_points(ctx, pgeo):
  """Generate a table of all the start points in the sequence."""
  
  logger.debug('start_points begin')

  sp = pd.read_sql("SELECT block_order, bf_uid, edge_order, lb_uid, source_x, source_y, arc_side, startnodenumber, ngd_uid, lu_uid FROM ordered_sequence WHERE edge_order = 1", ctx.obj['dest_db'])
  # seems like a waste of time - maybe rename the columns?
  sp['x'] = sp['source_x']
  sp['y'] = sp['source_y']

  # add a t_flag field
  sp['t_flag'] = None

  # add the ngd_uid - just pulled it from the sql query, since it's on the edge sequence table already

  # set the LUID (LU_UID was brought in from SQL)
  lu_info = pd.read_sql("SELECT luid, lu_uid FROM {}".format(pgeo), con=ctx.obj['src_db'])
  sp['luid'] = lu_info.loc[lu_info['lu_uid'] == sp['lu_uid']]

  sp.to_sql('start_points', con=ctx.obj['dest_db'])

  logger.debug('start_points end')

@click.command()
@click.argument('pgeo', envvar='SEQ_PARENT_LAYER')
@click.argument('roads', envvar='SEQ_ROAD_LAYER')
@click.pass_context
def t_intersections(ctx, pgeo, roads):
  """Find places where the road network forms a T intersection with the parent geography boundary."""

  logger.debug("t_intersections start")

  poly = gpd.GeoDataFrame.from_postgis("SELECT geom, ngd_uid FROM {}".format(pgeo), con=ctx.obj['src_db'])
  roads = gpd.GeoDataFrame.from_postgis("SELECT geom, ngd_uid FROM {}".format(roads), con=ctx.obj['src_db'])

  # needs to test the boundary of the polygon, not the polygon itself
  poly_edges = list(poly.geometry.boundary)
  roads['is_t'] = roads[roads['SGMNT_TYP_CDE'] == 2].geometry.apply(lambda r: forms_t(r, poly_edges))

  roads[roads['is_t'] == True].to_sql('t_intersection', con=ctx.obj['dest_db'])

  logger.debug('t_intersections end')

def forms_t(arc, edges):
  """Check if arc forms a T intersection with the provided edge list."""

  is_t = False
  for edge in edges:
    if arc.touches(edge):
      is_t = True
      # bail on match, no point in finding more
      break
  
  return is_t