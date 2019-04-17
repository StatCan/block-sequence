import logging
import os
from pathlib import Path

import click
import geopandas as gpd
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

logger = logging.getLogger()

@click.command()
@click.argument('parent_layer', envvar='SEQ_PARENT_LAYER')
@click.argument('parent_uid', envvar='SEQ_PARENT_UID')
@click.pass_context
def sp_weights(ctx, parent_layer, parent_uid):
  """Assess the weight of each node in an parent geography.
  
  Parent geography boundaries are used to calculate potential start points for routing through
  the road network. The more popular a given point, the less an enumerator would need to travel between 
  geographies, making it more desirable to use.
  """

  logger.debug("sp_weights start")

  # get every parent geography in the dataset
  sql = "SELECT {}, geom FROM {}".format(parent_uid, parent_layer)
  pgeo = gpd.GeoDataFrame.from_postgis(sql, ctx.obj['src_db'])
  # pull out the nodes in the polygons
  pgeo['coords'] = pgeo.geometry.boundary.apply(lambda x: x[0].coords)
  sp = pgeo[[parent_uid, 'coords']]
  d = []
  for r in sp.iterrows():
    k = r[1][0]
    v = r[1][1]
    for i in v:
      d.append((k,i))
  coord_pop = pd.DataFrame(d, columns=[parent_uid, 'node'])
  coord_pop['weight'] = coord_pop.groupby(['node'])[parent_uid].transform('count')
  
  # write it all to sqlite for reference by later steps
  coord_pop.to_sql('node_weights', con=ctx.obj['dest_db'])

  logger.debug("sp_weights end")

def get_circuit_distance(circuit, length_field):
  """Compute the total distance for a complete eulerian circuit."""
  
  return sum([edge[2][0][length_field] for edge in circuit])

def get_graph_distance(g, length_field):
  """Compute the total distance for a given graph."""

  return sum(nx.get_edge_attributes(g, weight_field_name).values())

def get_edge_count(g):
  """Calculate the total number of edges in a graph."""
  return len(g.edges())

def get_node_count(g):
  """Calculate the total number of nodes in a graph."""
  return len(g.nodes())

@click.command()
@click.argument('cid', envvar='SEQ_CHILD_UID', help='Field name of the child geography unique identifer')
@click.pass_context
def order_blocks(ctx, cid):
  """Calculate the block ordering based on the edge sequence."""

  # pull the edge sequence out of the database
  edge_sequence = pd.read_sql("SELECT * FROM edge_sequence", con=ctx.obj['src_db'])

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