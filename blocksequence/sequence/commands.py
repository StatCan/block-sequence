from ast import literal_eval
import itertools
import logging
from multiprocessing.dummy import Pool
import os
from pathlib import Path
import sys

import click
import networkx as nx
import numpy as np
import pandas as pd
import psycopg2
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.sql import select
from sqlalchemy.orm import Session, sessionmaker, scoped_session
from sqlalchemy.pool import SingletonThreadPool

logger = logging.getLogger(__name__)

@click.command()
@click.argument('bf_tbl', envvar='SEQ_BF_TABLE')
@click.argument('weight_field', envvar='SEQ_ROAD_COST')
@click.argument('parent_geo', envvar='SEQ_PARENT_LAYER')
@click.argument('parent_geo_uid', envvar='SEQ_PARENT_UID')
@click.option('--pid', default=None, show_default=True, help="Parent geography UID")
@click.option('--node_limit', type=int, default=5, show_default=True, help="How many potential nodes to try as start points")
@click.pass_context
def sequence(ctx, bf_tbl, weight_field, parent_geo, parent_geo_uid, pid, node_limit):
  """Sequence one or all LUs in the source data.

  If no pid is specified, all geographies found in the parent geography will be sequenced.
  """

  logger.debug("sequence start")

  # placeholder variables
  meta = MetaData()
  saved_geos = []

  # avoid lots of work if we're only going to do one parent geography
  src_url = ctx.obj['src_db'].url
  dest_url = ctx.obj['dest_db'].url
  if pid:
    logger.info("Limiting analysis to %s %s", parent_geo_uid, pid)
    saved_geos = sequence_geo(src_url, dest_url, bf_tbl, parent_geo_uid, pid, weight_field, node_limit)
  else:
    logger.debug("Getting list of parent geographies")

    # create a session on the DB to get the list of geographies
    Session = scoped_session(sessionmaker(bind=ctx.obj['src_db']))
    session = Session()

    edge_table = Table(bf_tbl, meta, autoload=True, autoload_with=ctx.obj['src_db'])

    # process the parent geographies one at a time
    # grabbing every parent geo at once could create memory issues on the system, so they are grabbed only 
    # as required.
    geos = []
    for pgeo in session.query(edge_table.c[parent_geo_uid]).distinct():
      # queries return a tuple representing the row. Grab the geo uid from the first column
      geos.append((src_url, dest_url, bf_tbl, parent_geo_uid, pgeo[0], weight_field, node_limit))
    
    with Pool() as p:
      saved_geos = p.starmap(sequence_geo, geos)

  # # leverage pandas to write all the edge data to the database
  # logger.debug("Generating dataframe for output")
  # edge_sequence = pd.DataFrame.from_records(edgelist)
  # # don't waste time sorting - the DB can do that
  # # edge_sequence.sort_values(by='sequence', inplace=True)
  
  # # write the edge list to the outputs db
  # logger.info("Writing sequence results to database")
  # edge_sequence.to_sql('edge_sequence', con=ctx.obj['dest_db'], if_exists='replace', index=False)
  logger.info('Processed %s geographies', len(saved_geos))

  logger.debug('sequence end')

def sequence_geo(src_url, dest_url, bf_tbl, parent_geo_uid, pid, weight_field, node_limit):
  """Sequence a particular parent level geography."""

  logger.debug('sequence_geo start')

  # sqlalchemy setup
  meta = MetaData()
  src_db = create_engine(src_url)
  SRCSession = scoped_session(sessionmaker(bind=src_db))
  src_session = SRCSession()
  dest_db = create_engine(dest_url, poolclass=SingletonThreadPool)
  DESTSession = scoped_session(sessionmaker(bind=dest_db))
  dest_session = DESTSession()

  logger.info("Sequencing %s %s", parent_geo_uid, pid)

  # grab a copy of the edge table for this geography
  edge_table = Table(bf_tbl, meta, autoload=True, autoload_with=src_db)
  edge_table_query = select([edge_table], edge_table.c[parent_geo_uid] == pid)
  all_edges = pd.read_sql(edge_table_query, con=src_db, coerce_float=False)
  # prevent pandas from storing numeric IDs as floats to avoid matching issues later
  all_edges[parent_geo_uid] = all_edges[parent_geo_uid].astype(int)

  # build a graph from the edge list in the DataFrame
  g = nx.convert_matrix.from_pandas_edgelist(all_edges, 'start_node', 'end_node', True, nx.MultiGraph)
  logger.debug("MultiGraph with %s nodes and %s edges built for %s", len(g.nodes()), len(g.edges()), pid)

  # ensure the graph is connected, otherwise it can't be made into a eulerian circuit
  is_connected = nx.is_connected(g)
  logger.debug("%s graph is fully connected: %s", pid, is_connected)
  if not is_connected:
    logger.error("Disconnected graph found for %s. Sequencing subgraphs.", pid)
    all_edges = pd.concat([sequence_edges(g.subgraph(comp), dest_db, parent_geo_uid, pid, weight_field, node_limit) for comp in nx.connected_components(g)])
    # for comp in nx.connected_components(g):
    #   g_sub = g.subgraph(comp)
    #   edgelist.append(sequence_edges(g_sub, dest_db, parent_geo_uid, pid, weight_field, node_limit))
  else:
    all_edges = sequence_edges(g, dest_db, parent_geo_uid, pid, weight_field, node_limit)

  all_edges['edge_order'] = all_edges.sort_values('seq').groupby('block', sort=False).cumcount()+1
  all_edges['block_order'] = all_edges.sort_values('seq').groupby('block', sort=False).ngroup()+1
  all_edges['chain_id'] = np.where(all_edges['eo'] == 1, 1, 0)

  # write the edge list to the outputs db
  logger.info("Writing %s %s sequence results to database", parent_geo_uid, pid)
  edge_sequence.to_sql('edge_sequence', con=dest_db, if_exists='append', index=False)

  logger.debug('sequenc_geo end')
  return pid


def sequence_edges(g, dest_db, parent_geo_uid, pid, weight_field, node_limit):
  """Generate a sequence of edges for the given graph."""

  logger.debug("sequence_edges start")

  meta = MetaData()
  DESTSession = scoped_session(sessionmaker(bind=dest_db))
  dest_session = DESTSession()

  # find nodes of odd degree (dead ends)
  logger.info("Finding nodes of odd degree in graph for %s", pid)
  nodes_odd_degree = [v for v,d in g.degree() if d % 2 == 1]
  logger.debug("Found %s nodes of odd degree in %s", len(nodes_odd_degree), pid)

  # compute pairs for odd degree nodes to get out of dead ends
  logger.info("Calculating node pairs for odd degree nodes in %s", pid)
  odd_node_pairs = list(itertools.combinations(nodes_odd_degree, 2))
  logger.debug("Calculated %s node pairs in %s", len(odd_node_pairs), pid)
  if len(odd_node_pairs) > 150:
    logger.warning("%s node pairs is high, and will be slower to process", len(odd_node_pairs))
  
  # compute shortest paths
  logger.info("Calculating shortest paths for odd node pairs in %s", pid)
  odd_node_shortest_paths = get_shortest_paths_distances(g, odd_node_pairs, weight_field)

  # create a complete graph
  logger.info("Making graph for %s with shortest paths for odd nodes", pid)
  g_odd_complete = create_complete_graph(odd_node_shortest_paths)

  # compute minimum weight matches
  logger.info("Calculating minimum weight matches for odd node pairs in %s", pid)
  odd_matching_dupes = nx.algorithms.max_weight_matching(g_odd_complete, True)
  odd_matching = list(pd.unique([tuple(sorted([k,v])) for k, v in odd_matching_dupes]))
  logger.debug("Number of edges to augment original graph %s with: %s", pid, len(odd_matching))

  # augment the original graph with the new values
  logger.info("Augmenting original graph %s with deduplicated odd node pairs", pid)
  g_aug = add_augmented_path_to_graph(g, odd_matching)
  logger.debug("Edge count in %s augmented graph: %s", pid, len(g_aug.edges()))

  # validate that all nodes are now of even degree
  logger.info("Validating that all %s nodes are of even degree", pid)
  odd_values = [v for v,d in g_aug.degree() if d % 2 == 1]
  if odd_values:
    logger.error("Odd degree values found in %s, exiting", pid)
    return []
  
  # to aide in debugging, dump the current node list
  logger.debug("All nodes in %s geography:\n %s", pid, g.nodes())
  
  # pull popular nodes from the 'node_weights' table
  logger.debug("Finding weighted nodes for parent geography %s", pid)
  node_weights = Table('node_weights', meta, autoload=True, autoload_with=dest_db)
  # pg_node_query = select([node_weights])
  # pg_node_conn = dest_session.query(node_weights).where(node_weights.c.lu_uid == pid).order_by(node_weights.c.weight.desc()).limit(node_limit).all()
  # pg_node_result = pg_node_conn.execute(pg_node_query)
  pg_nodes = dest_session.query(node_weights).\
    filter(node_weights.c.lu_uid == pid).\
    order_by(node_weights.c.weight.desc()).\
    limit(node_limit).\
    all()
  DESTSession.remove()
  logger.debug("Calculating best route from %s nodes in %s", len(pg_nodes), pid)

  # iterate the start points, calculating the circuit
  shortest_distance = -1
  chosen_circuit = None
  for point in pg_nodes:
    start_point = point[1]

    # make sure this node exists in the node list
    if not start_point in g.nodes():
      logger.warning("Node %s not found in %s graph, skipping.", start_point, pid)
      continue

    # calculate the eulerian circuit
    # start and end point are assumed to be the same, which may not be the optimal choice
    # test the total travel distance of the circuit for multiple start points
    logger.debug("Computing eulerian circuit for %s")
    euler_circuit = create_eulerian_circuit(g_aug, g, weight_field, start_point)

    # total distance
    circuit_dist = sum([edge[2][0][weight_field] for edge in euler_circuit])
    logger.info("Circuit distance for %s from %s: %s", pid, start_point, circuit_dist)
    if circuit_dist < shortest_distance or shortest_distance == -1:
      shortest_distance = circuit_dist
      chosen_circuit = euler_circuit
      logger.debug("%s is now shortest distance in %s", start_point, pid)
  
  # make sure the graph was actually calculated
  if shortest_distance == -1:
    logger.critical("No possible circuit found for %s %s using %s start nodes", parent_geo_uid, pid, node_limit)
    return pd.DataFrame()
  
  # set the sequence on the graph
  for i, e in enumerate(nx.eulerian_circuit(g)):
    set_edge_sequence(g, e, i)
  
  return nx.to_pandas_edgelist(g)
  
  # find the circuit with the shortest distance
  graph_distance = sum(nx.get_edge_attributes(g, weight_field).values())
  logger.info("{0} Circuit distance: {1:.2f}".format(pid, shortest_distance))
  logger.info("{0} Graph distance: {1:.2f}".format(pid, graph_distance))
  logger.info("{0} Solution is {1:.2f}% efficient".format(pid, (graph_distance / shortest_distance) * 100))

  # use the chosen circuit to generate an edge list
  logger.debug("Getting edge list for %s %s circuit", parent_geo_uid, pid)
  cpp_edgelist = create_cpp_edgelist(chosen_circuit)

  # flatten an edge list into a final sequence and write to db
  logger.debug("Flattening edge list for %s into final sequence", pid)
  flat_edgelist = flatten_edgelist(cpp_edgelist)

  logger.debug("sequence_edges end")
  return flat_edgelist

def set_edge_sequence(graph, edge, seq):
  """Set a sequence value on the given graph edge."""

  # get the number of edges in this set
  sides = graph.number_of_edges(edge[0], edge[1])

  # iterate the edges, marking the first one we see with the sequence
  for s in range(sides):
    # get the data for this edge
    data = graph.get_edge_data(edge[0],edge[1],s)

    # if this one has already been seen, skip it
    if 'seq' in data:
      continue

    # set the sequence on this edge
    graph[edge[0]][edge[1]][s]['seq'] = seq

    # after marking an edge, bail to avoid putting the same sequence on more than one edge
    return

def get_shortest_paths_distances(graph, pairs, edge_weight_name):
  """Compute the shortest distance between each pair of nodes in a graph.
  
  Returns a dictionary keyed on node pairs (tuples).
  """

  logger.debug("get_shortest_paths_distances start")
  distances = {}
  for pair in pairs:
    length = nx.dijkstra_path_length(graph, pair[0], pair[1], weight=edge_weight_name)
    distances[pair] = length
    logger.debug("Found distance of %s for pair %s", length, pair)

  logger.debug("get_shortest_paths_distances end")
  return distances


def create_complete_graph(pair_weights, flip_weights=True):
  """Create a complete graph from a set of weighted pairs."""

  logger.debug("create_complete_graph start")
  g = nx.Graph()
  for k, v in pair_weights.items():
    wt_i = - v if flip_weights else v
    g.add_edge(k[0], k[1], **{'distance': v, 'weight': wt_i})
    logger.debug("Added edge %s <-> %s", k[0], k[1])
  
  logger.debug("create_complete_graph end")
  return g


def add_augmented_path_to_graph(graph, min_weight_pairs):
  """Add the min weight matching edges to the original graph.
  Parameters:
    graph: NetworkX graph
    min_weight_pairs: list[tuples] of node pairs from min weight matching
  Returns:
    augmented NetworkX graph
  """
  
  logger.debug("add_augmented_path_to_graph start")
  # use a MultiGraph to allow for parallel edges
  graph_aug = nx.MultiGraph(graph.copy())
  for pair in min_weight_pairs:
    logger.debug("Augmenting graph with %s <-> %s", pair[0], pair[1])
    graph_aug.add_edge(pair[0],
                      pair[1],
                      **{'distance': nx.dijkstra_path_length(graph, pair[0], pair[1]),
                        'bf_type': 'augmented'}
                      )
  
  logger.debug("add_augmented_path_to_graph end")
  return graph_aug


def create_cpp_edgelist(euler_circuit):
  """
  Create the edgelist.

  Combine duplicate edges and keep track of their sequence and number of walks.

  Parameters:
      euler_circuit: list[tuple] from create_eulerian_circuit
  """

  logging.debug("create_cpp_edgelist start")
  cpp_edgelist = {}
  preferred_side_field = os.getenv('SEQ_BF_SIDE_FIELD')
  preferred_side_value = os.getenv('SEQ_BF_SIDE_PREFERRED')

  for i, e in enumerate(euler_circuit):
    logger.debug("processing edge %s", i)
    edge   = frozenset([e[0], e[1]])
    
    # each edge can have multiple paths (L/R), so number accordingly
    if edge not in cpp_edgelist:
      cpp_edgelist[edge] = e
      # label the right edge with the sequence number
      # this implements the 'right hand rule'
      for j, bf in enumerate(cpp_edgelist[edge][2]):
        if cpp_edgelist[edge][2][j][preferred_side_field] == preferred_side_value:
          cpp_edgelist[edge][2][j]['sequence'] = i
          if 'visits' in cpp_edgelist[edge][2][j]:
            cpp_edgelist[edge][2][j]['visits'] += 1
          else:
            cpp_edgelist[edge][2][j]['visits'] = 1
    else:
      # label the other edge with a sequence number
      for j, bf in enumerate(cpp_edgelist[edge][2]):
        if not cpp_edgelist[edge][2][j].get('sequence'):
          cpp_edgelist[edge][2][j]['sequence'] = i
          if 'visits' in cpp_edgelist[edge][2][j]:
            cpp_edgelist[edge][2][j]['visits'] += 1
          else:
            cpp_edgelist[edge][2][j]['visits'] = 1
          continue

  logging.debug("create_cpp_edgelist end")
  return list(cpp_edgelist.values())


def flatten_edgelist(edgelist):
  """Turn a MultiGraph edge list into a flattened list."""

  logging.debug("flatten_edgelist start")

  for multiedge in edgelist:
    source = literal_eval(multiedge[0])
    target = literal_eval(multiedge[1])
    logger.debug("Flatting edge %s -> %s", source, target)

    for edge in multiedge[2]:
      edge_attribs = multiedge[2][edge]
      edge_attribs['source_x'] = source[0]
      edge_attribs['source_y'] = source[1]
      edge_attribs['target_x'] = target[0]
      edge_attribs['target_y'] = target[1]
      # logger.debug("New edge: %s", edge_attribs)
      yield edge_attribs
  
  logging.debug("flatten_edgelist end")


def create_eulerian_circuit(graph_augmented, graph_original, weight_field_name, start_node=None):
  """Create the eulerian path using only edges from the original graph."""

  logging.debug("create_eulerian_circuit start")

  euler_circuit = []

  logger.debug("Building naive circuit through augmented graph from %s", start_node)
  naive_circuit = list(nx.eulerian_circuit(graph_augmented, source=start_node))
  logger.debug("Naive circuit has %s edges", len(naive_circuit))
  
  for edge in naive_circuit:
    # get the original edge data
    edge_data = graph_augmented.get_edge_data(edge[0], edge[1])
    
    # this is not an augmented path, just append it to the circuit
    if edge_data[0].get('bf_type') != 'augmented':
      # logger.debug("%s is not augmented, keeping in the circuit", edge_data)
      edge_att = graph_original[edge[0]][edge[1]]
      # appends a tuple to the final circuit
      euler_circuit.append((edge[0], edge[1], edge_att))
      continue
  
    # edge is augmented, find the shortest 'real' route
    logger.debug("Augmented path found, calculating shortest path between nodes")
    aug_path = nx.shortest_path(graph_original, edge[0], edge[1], weight=weight_field_name)
    aug_path_pairs = list(zip(aug_path[:-1], aug_path[1:]))

    logging.debug('Filling in edges for augmented edge: %s', edge)

    # add the edges from the shortest path
    for edge_aug in aug_path_pairs:
        edge_aug_att = graph_original[edge_aug[0]][edge_aug[1]]
        euler_circuit.append((edge_aug[0], edge_aug[1], edge_aug_att))
  
  logger.debug("Eulerian circuit has %s edges", len(euler_circuit))
  
  logging.debug("create_eulerian_circuit end")
  return euler_circuit
