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
  pgeo_table = Table(parent_layer, meta, autoload=True, autoload_with=ctx.obj['db'])
  pgeo_select = select([pgeo_table.c[parent_uid], pgeo_table.c.geom])

  # get all the boundaries from the parent geography and grab the node coordinates
  pgeo = (gpd.GeoDataFrame.from_postgis(pgeo_select, ctx.obj['db'])
            .pipe(get_coords)
            .drop('geom', axis=1))

  logger.debug("Grouping nodes to determine popularity")
  pgeo['weight'] = pgeo.groupby(['node'])[parent_uid].transform('count')

  # cast the node to str for writing to the db
  pgeo['node'] = pgeo['node'].astype(str)
  
  # write it all to for reference by later steps
  logger.debug("Saving to node_weights table")
  pgeo.to_sql('node_weights', con=ctx.obj['db'], if_exists='replace', index=False)

  logger.debug("node_weights end")

def get_coords(df):
  """Pull the coordinate value out from each node in a polygon geometry."""
  df['node'] = df.geometry.boundary.apply(lambda x: round_coords(x[0].coords))
  return df

def round_coords(coord_pair, precision=5):
  x = round(coord_pair[0], precision)
  y = round(coord_pair[1], precision)
  return (x,y)

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