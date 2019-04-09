import itertools
import logging
import os
from pathlib import Path

import click
import networkx as nx
import pandas as pd
import psycopg2

logger = logging.getLogger()

@click.command()
@click.option('--pid', default=None, show_default=True, help="Parent geography UID")
@click.option('--weight_field', default='SHAPE_Length')
@click.pass_context
def sequence(ctx, pid, weight_field):
  """Sequence one or all LUs in the source data.

  If no pid is specified, all geographies found in the parent geography will be sequenced.
  """

  logger.debug("sequence started")

  # build a DataFrame from the database
  lu_edges = None #sql results as dataframe

  # build a graph from the edge list in the DataFrame
  g = nx.convert_matrix.from_pandas_edgelist(lu_edges, 'start_node', 'end_node', True, nx.MultiGraph)
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

  # TODO: calculate the start points

  # calculate the eulerian circuit
  # start and end point are assumed to be the same, which may not be the optimal choice
  # test the total travel distance of the circuit for multiple start points

  # TODO: need algorithm to pick "best" route in start point popularity vs total length



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
