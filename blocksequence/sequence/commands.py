import itertools
import logging
import os
from pathlib import Path

import click
import networkx as nx
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.sql import select
from ast import literal_eval

logger = logging.getLogger(__name__)

@click.command()
@click.argument('bf_tbl', envvar='SEQ_BF_TABLE')
@click.argument('weight_field', envvar='SEQ_ROAD_COST')
@click.argument('parent_geo', envvar='SEQ_PARENT_LAYER')
@click.argument('parent_geo_uid', envvar='SEQ_PARENT_UID')
@click.option('--pid', default=None, show_default=True, help="Parent geography UID")
@click.pass_context
def sequence(ctx, bf_tbl, weight_field, parent_geo, parent_geo_uid, pid):
  """Sequence one or all LUs in the source data.

  If no pid is specified, all geographies found in the parent geography will be sequenced.
  """

  logger.debug("sequence start")

  # hold onto sqlalchemy metadata
  meta = MetaData()

  # build a DataFrame from the database
  if pid:
    sql = "SELECT * FROM {} WHERE {}={}".format(bf_tbl, parent_geo, pid)
  else:
    sql = "SELECT * FROM {}".format(bf_tbl)
  all_edges = pd.read_sql(sql, con=ctx.obj['src_db'])

  # group the edges by parent geo UID
  pg_grouped = all_edges.groupby(parent_geo)
  for pg_uid, pg_group in pg_grouped:
    logger.debug("Working on parent geography: %s", pg_uid)

    # build a graph from the edge list in the DataFrame
    g = nx.convert_matrix.from_pandas_edgelist(pg_group, 'start_node', 'end_node', True, nx.MultiGraph)
    logger.debug("Multigraph with %s nodes and %s edges build", len(g.nodes()), len(g.edges()))

    # ensure the graph is connected, otherwise it can't be made into a eulerian circuit
    is_connected = nx.is_connected(g)
    logger.debug("Graph is fully connected: %s", is_connected)

    # find nodes of odd degree (dead ends)
    logger.info("Finding nodes of odd degree in graph")
    nodes_odd_degree = [v for v,d in g.degree() if d % 2 == 1]
    logger.debug("Found %s nodes of odd degree", len(nodes_odd_degree))

    # compute pairs for odd degree nodes to get out of dead ends
    logger.info("Calculating node pairs for odd degree nodes")
    odd_node_pairs = list(itertools.combinations(nodes_odd_degree, 2))
    logger.debug("Calculated %s node pairs", len(odd_node_pairs))
    if len(odd_node_pairs) > 150:
      logger.warning("%s node pairs is high, and will be slower to process", len(odd_node_pairs))
    
    # compute shortest paths
    logger.info("Calculating shortest paths for odd node pairs")
    odd_node_shortest_paths = get_shortest_paths_distances(g, odd_node_pairs, weight_field)

    # create a complete graph
    logger.info("Making graph with shortest paths for odd nodes")
    g_odd_complete = create_complete_graph(odd_node_shortest_paths)

    # compute minimum weight matches
    logger.info("Calculating minimum weight matches for odd node pairs")
    odd_matching_dupes = nx.algorithms.max_weight_matching(g_odd_complete, True)
    odd_matching = list(pd.unique([tuple(sorted([k,v])) for k, v in odd_matching_dupes]))
    logger.debug("Number of edges to augment original graph with: %s", len(odd_matching))

    # augment the original graph with the new values
    logger.info("Augmenting original graph with deduplicated odd node pairs")
    g_aug = add_augmented_path_to_graph(g, odd_matching)
    logger.debug("Edge count in augmented graph: %s", len(g_aug.edges()))

    # TODO: validate that all nodes are now of even degree
    # pd.value_counts([e[1] for e in g_aug.degree()])

    # pull popular nodes from the 'node_weights' table
    node_weights = Table('node_weights', meta, autoload=True, autoload_with=ctx.obj['dest_db'])
    pg_nodes = select([node_weights]).where(node_weights.c.lu_uid == pg_uid).order_by(node_weights.c.weight.desc()).limit(5)

    # iterate the start points, calculating the circuit
    shortest_distance = -1
    chosen_circuit = None
    for point in pg_nodes:
      start_point = literal_eval(point[1])

      # calculate the eulerian circuit
      # start and end point are assumed to be the same, which may not be the optimal choice
      # test the total travel distance of the circuit for multiple start points
      logger.debug("Computing eulerian circuit")
      euler_circuit = create_eulerian_circuit(g_aug, g, weight_field, start_point)

      # total distance
      circuit_dist = sum([edge[2][0][weight_field] for edge in euler_circuit])
      logger.info("Circuit distance from %s: %s", start_point, circuit_dist)
      if circuit_dist < shortest_distance or shortest_distance == -1:
        shortest_distance = circuit_dist
        chosen_circuit = euler_circuit
        logger.debug("%s has shortest distance", start_point)
      
    # find the circuit with the shortest distance
    graph_distance = sum(nx.get_edge_attributes(g, weight_field).values())
    logger.info("Circuit distance: {0:.2f}".format(shortest_distance))
    logger.info("Graph distance: {0:.2f}".format(graph_distance))
    logger.info("Solution is {0:.2f}% efficient".format(graph_distance / chosen_circuit))

    # use the chosen circuit to generate an edge list
    logger.debug("Getting edge list for circuit")
    cpp_edgelist = create_cpp_edgelist(chosen_circuit)

    # flatten an edge list into a final sequence and write to db
    logger.debug("Flattening edge list into final sequence")
    edge_sequence = pd.DataFrame.from_records(flatten_edgelist(cpp_edgelist))
    edge_sequence.sort_values(by='sequence', inplace=True)
    
    # write the edge list to the outputs db
    edge_sequence.to_sql('edge_sequence', con=ctx.obj['dest_db'])

    logger.debug('sequence end')


def get_shortest_paths_distances(graph, pairs, edge_weight_name):
  """Compute the shortest distance between each pair of nodes in a graph.
  
  Returns a dictionary keyed on node pairs (tuples).
  """

  logger.debug("get_shortest_paths_distances start")
  distances = {}
  for pair in pairs:
    distances[pair] = nx.dijkstra_path_length(graph, pair[0], pair[1], weight=edge_weight_name)

  logger.debug("get_shortest_paths_distances end")
  return distances


def create_complete_graph(pair_weights, flip_weights=True):
  """Create a complete graph from a set of weighted pairs."""

  logger.debug("create_complete_graph start")
  g = nx.Graph()
  for k, v in pair_weights.items():
    wt_i = - v if flip_weights else v
    g.add_edge(k[0], k[1], **{'distance': v, 'weight': wt_i})
  
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
    graph_aug.add_edge(pair[0],
                      pair[1],
                      **{'distance': nx.dijkstra_path_length(graph, pair[0], pair[1]),
                        'bf_type': 'augmented'}
                      )
  
  logger.debug("add_augmented_path_to_graph end")
  return graph_aug


def create_cpp_edgelist(euler_circuit):
  """
  Create the edgelist without parallel edge for the visualization
  Combine duplicate edges and keep track of their sequence and # of walks
  Parameters:
      euler_circuit: list[tuple] from create_eulerian_circuit
  """

  logging.debug("create_cpp_edgelist start")
  cpp_edgelist = {}

  for i, e in enumerate(euler_circuit):
    edge   = frozenset([e[0], e[1]])
    
    # each edge can have multiple paths (L/R), so number accordingly
    if edge not in cpp_edgelist:
      cpp_edgelist[edge] = e
      # label the right edge with the sequence number
      # this implements the 'right hand rule'
      for j, bf in enumerate(cpp_edgelist[edge][2]):
        if cpp_edgelist[edge][2][j]['ARC_SIDE'] == 'R':
          cpp_edgelist[edge][2][j]['sequence'] = i
          cpp_edgelist[edge][2][j]['visits'] = 1 # shouldn't be hardcoded
    else:
      # label the other edge with a sequence number
      for j, bf in enumerate(cpp_edgelist[edge][2]):
        if not cpp_edgelist[edge][2][j].get('sequence'):
          cpp_edgelist[edge][2][j]['sequence'] = i
          cpp_edgelist[edge][2][j]['visits'] = 1 # shouldn't be hardcoded
          continue

  logging.debug("create_cpp_edgelist end")
  return list(cpp_edgelist.values())


def flatten_edgelist(edgelist):
  """Turn a MultiGraph edge list into a flattened list."""

  logging.debug("flatten_edgelist start")
  for multiedge in edgelist:
    source = multiedge[0]
    target = multiedge[1]
    for edge in multiedge[2]:
      edge_attribs = multiedge[2][edge]
      edge_attribs['source_x'] = source[0]
      edge_attribs['source_y'] = source[1]
      edge_attribs['target_x'] = target[0]
      edge_attribs['target_y'] = target[1]
      yield edge_attribs
  
  logging.debug("flatten_edgelist end")


def create_eulerian_circuit(graph_augmented, graph_original, weight_field_name, start_node=None):
  """Create the eulerian path using only edges from the original graph."""

  logging.debug("create_eulerian_circuit start")
  euler_circuit = []
  naive_circuit = list(nx.eulerian_circuit(graph_augmented, source=start_node))
  
  for edge in naive_circuit:
    # get the original edge data
    edge_data = graph_augmented.get_edge_data(edge[0], edge[1])
    
    # this is not an augmented path, just append it to the circuit
    if edge_data[0].get('trail') != 'augmented':
      edge_att = graph_original[edge[0]][edge[1]]
      # appends a tuple to the final circuit
      euler_circuit.append((edge[0], edge[1], edge_att))
      continue
  
    # edge is augmented, find the shortest 'real' route
    aug_path = nx.shortest_path(graph_original, edge[0], edge[1], weight=weight_field_name)
    aug_path_pairs = list(zip(aug_path[:-1], aug_path[1:]))

    logging.debug('Filling in edges for augmented edge: %s', edge)

    # add the edges from the shortest path
    for edge_aug in aug_path_pairs:
        edge_aug_att = graph_original[edge_aug[0]][edge_aug[1]]
        euler_circuit.append((edge_aug[0], edge_aug[1], edge_aug_att))
  
  logging.debug("create_eulerian_circuit end")
  return euler_circuit
