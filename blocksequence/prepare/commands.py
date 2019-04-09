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
@click.option('--parent_layer', help="Parent geography layer name")
@click.option('--parent_uid', help="Parent geography layer UID field name")
@click.pass_context
def sp_weights(ctx, parent_layer, parent_uid):
  """Assess the weight of each node in an LU boundary to calculate preferred start points for 
  distance calculations.
  """

  logger.debug("sp_weights start")

  # for each geometry, get the boundary of objects down to coordinate values and build a 
  # dataframe that value_counts() can be run on to find the ranking
  source_db = ctx['source_db']
  source_user = ctx['source_user']
  source_pass = ctx['source_pass']
  source_host = ctx['source_host']
  conn = psycopg2.connect(database=source_db, user=source_user, password=source_pass, host=source_host)
  sql = "SELECT {}, geometry FROM {}".format(parent_uid, parent_layer)

  pgeo = gpd.GeoDataFrame.from_postgis(sql, conn)
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
  output = ctx['output']
  engine = create_engine('sqlite://{}'.format(output), echo=False)
  coord_pop.to_sql('node_weights', con=engine)

  logger.debug("sp_weights end")
