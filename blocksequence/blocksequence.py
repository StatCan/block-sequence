import collections
import itertools
import logging

import networkx as nx
import pandas as pd

class BlockSequence:

    def __init__(self, edges, source='source', target='target', weight='length'):
        """Initialize a new block sequence object on the provided edge DataFrame."""
        self.edges = edges
        self.source_field = source
        self.target_field = target
        self.weight_field = weight

        # the type of graph to be used in generating the sequence
        self.graph_type = nx.MultiGraph

        # how to identify augmented edges
        self.augmented_field_name = 'bf_type'
        self.augmented_field_value = 'augmented'

        # the field name the sequence is calculated in
        self.sequence_field_name = 'sequence'

        self.graph = self._create_graph()

    def _create_graph(self):
        """Create a graph from the edge list."""

        return nx.from_pandas_edgelist(self.edges,
                                       self.source_field,
                                       self.target_field,
                                       edge_attr=True,
                                       create_using=self.graph_type)


    def _is_connected_graph(self):
        """Boolean check of if the graph is fully connected."""

        return nx.is_connected(self.graph)

    def _is_empty_graph(self):
        """Boolean check of if the graph is empty."""

        return nx.is_empty(self.graph)


    def eulerian_circuit(self, block_field, drop_augmented=True, edge_field='', boundary_attr=None):
        """Produce a eulerian circuit through the edge list, providing both a block and edge order."""

        # empty graphs cannot be eulerian, so just return an empty dataframe
        if self._is_empty_graph():
            return pd.DataFrame()

        # graphs can be disconnected, so cycle over the components from largest to smallest
        # connected graphs will only have one component
        all_components = []
        for component in sorted(nx.connected_components(self.graph), key=len, reverse=True):
            subgraph = self.graph.subgraph(component)
            # generate a circuit for only this subgraph
            circuit = self._component_circuit(subgraph, boundary_attr)
            # append it to the list of components to be given a block and edge order
            all_components.append(circuit)

        # turn all the component dataframes into a single dataframe
        edge_sequence = pd.concat(all_components, sort=False, ignore_index=True)

        # remove any augmented edges, if the desired
        if drop_augmented:
            edge_sequence = edge_sequence.drop_duplicates(edge_field)

        # calculate the block and edge order
        edge_sequence = edge_sequence.pipe(self._calculate_block_order, [block_field])

        return edge_sequence


    def _component_circuit(self, graph, boundary_attr):
        """Generate a eulerian circuit on the supplied graph.

        Where a proper eulerian circuit cannot be calculated, augmented edges are added to allow for a complete
        circuit.
        """

        # find nodes of odd degree
        nodes_odd_degree = [v for v, d in graph.degree() if d % 2 == 1]

        # compute pairs for odd degree nodes to get out of dead ends
        # this generates a pairing of the dead ends so that shortest path routes can be found between them
        odd_node_pairs = list(itertools.combinations(nodes_odd_degree, 2))

        # for the odd node pairs, find the shortest path between them
        odd_node_shortest_paths = self._get_shortest_paths_distances(graph, odd_node_pairs)

        # create a complete graph from the shortest paths
        # this will be used to augment the original graph so that dead ends have a way out
        g_odd_complete = self._create_complete_graph(odd_node_shortest_paths)

        # compute minimum weight matches to find the 'best' routes between the dead ends
        # networkx doesn't have a minimum weight match algorithm, so just generate the maximum weight matching and reverse
        # the sorting on it
        odd_matching_dupes = nx.algorithms.max_weight_matching(g_odd_complete, True)
        odd_matching = list(pd.unique([tuple(sorted([k, v])) for k, v in odd_matching_dupes]))

        # augment the original graph with the edges calculated from the odd matching
        g_aug = self._add_augmented_path_to_graph(graph, odd_matching)

        # use the node with the most start points as the first place to start from
        start_nodes = self._get_preferred_start_nodes(graph, boundary_attr)

        # calculate the shortest circuit through the graph
        shortest_distance = -1
        chosen_circuit = None
        for start_node in start_nodes:
            full_circuit = self._create_eulerian_circuit(g_aug, graph, start_node)

            circuit_distance = self._calculate_circuit_distance(full_circuit)
            if circuit_distance < shortest_distance or shortest_distance == -1:
                shortest_distance = circuit_distance
                chosen_circuit = full_circuit

        # edges that are within the group field, but don't touch a boundary can result in not finding a suitable
        # start node, so one is just picked to get a result
        if shortest_distance == -1:
            alternate_start_points = sorted(dict(graph.degree()).items(), key=lambda kv: kv[1])
            chosen_start_point = alternate_start_points[0][0]
            full_circuit = self._create_eulerian_circuit(g_aug, graph, chosen_start_point)
            circuit_distance = self._calculate_circuit_distance(full_circuit)
            chosen_circuit = full_circuit

        # circuits are a list, so convert it back into a graph
        circuit_graph = nx.from_edgelist(chosen_circuit, create_using=self.graph_type)
        # generate a dataframe from the graph to be given back, sorted by the sequence
        circuit_df = nx.to_pandas_edgelist(circuit_graph).sort_values('sequence')

        return circuit_df


    def _calculate_block_order(self, df, group_field):
        """Calculate the block_order and edge_order values for the given dataframe.

        The dataframe is grouped by the group_field to calculate the order, but will not be sorted. It must be sorted
        before being passed in.
        """

        # group the data based on the supplied field
        block_group = df.groupby(group_field, sort=False)

        # calculate the edge order
        df['edge_order'] = block_group.cumcount() + 1

        # calculate the block order
        df['block_order'] = block_group.ngroup() + 1

        return df


    def _calculate_circuit_distance(self, circuit):
        """Calculate the total weight of the circuit based on the weight_field."""

        return sum([edge[2][self.weight_field] for edge in circuit])


    def _get_shortest_paths_distances(self, graph, pairs):
        """Compute the shortest distance between each pair of nodes in a graph.

            Returns a dictionary keyed on node pairs (tuples).
            """

        # generate a lookup dictionary of node pairs and their length through the graph
        distances = {}
        for pair in pairs:
            # get the shortest distance through the graph between the nodes
            length = nx.dijkstra_path_length(graph, pair[0], pair[1], weight=self.weight_field)
            distances[pair] = length

        return distances

    def _create_complete_graph(self, pair_weights, flip_weights=True):
        """Create a complete graph from a set of weighted pairs."""

        g = nx.Graph()
        # go through every node pair, creating a graph from the paths
        for k, v in pair_weights.items():
            # flip the weights so that the longest length has the lowest (negative) value
            wt_i = - v if flip_weights else v
            # add the edge to the graph, recording the length as the weight
            g.add_edge(k[0], k[1], **{'distance': v, 'weight': wt_i})

        return g

    def _add_augmented_path_to_graph(self, graph, min_weight_pairs):
        """Add the min weight matching edges to the original graph.
        Parameters:
          graph: NetworkX graph
          min_weight_pairs: list[tuples] of node pairs from min weight matching
        Returns:
          augmented NetworkX graph
        """

        # use a MultiGraph to allow for parallel edges
        # create a copy of the original graph so that things can be added to it
        graph_aug = nx.MultiGraph(graph.copy())
        for pair in min_weight_pairs:
            # add the edge to the graph, marking it as augmented
            graph_aug.add_edge(pair[0],
                               pair[1],
                               **{'distance': nx.dijkstra_path_length(graph, pair[0], pair[1]),
                                  self.augmented_field_name: self.augmented_field_value}
                               )

        return graph_aug

    def _get_preferred_start_nodes(self, graph, boundary_attr):
        """Generate a list of start points to be used in route evaluations."""

        # get the boundary indicator for every edge in the graph
        # this will drop any edges that don't have a boundary flag, but if input is properly configured that should be none
        edges_boundary_flagged = nx.get_edge_attributes(graph, boundary_attr)
        boundary_edges_only = [n for n in edges_boundary_flagged.keys() if edges_boundary_flagged.get(n) == 'true']
        boundary_nodes = []
        # this is a multigraph, so each edge is defined by the nodes plus a view index
        for n1, n2, index in boundary_edges_only:
            boundary_nodes.append(n1)
            boundary_nodes.append(n2)
        node_bunch = set(boundary_nodes)

        # sort the nodes by popularity
        node_list = sorted(dict(graph.degree(node_bunch)).items(), key=lambda kv: kv[1], reverse=True)
        # just need the node IDs, not the degree values
        node_list = [n[0] for n in node_list]

        return node_list

    def _create_eulerian_circuit(self, graph_augmented, graph_original, start_node=None):
        """Create the eulerian path using only edges from the original graph."""

        euler_circuit = []

        # for a naive circuit from the augmented graph
        naive_circuit = list(nx.eulerian_circuit(graph_augmented, source=start_node, keys=True))

        circuit_sequence = 0
        for edge in naive_circuit:
            # get the original edge data
            edge_data = graph_augmented.get_edge_data(edge[0], edge[1], key=edge[2])

            # this is not an augmented path, just append it to the circuit (it's part of the original BF graph)
            if edge_data.get(self.augmented_field_name) != self.augmented_field_value:
                # logger.debug("%s is not augmented, keeping in the circuit", edge_data)
                edge_att = edge_data
                edge_att[self.sequence_field_name] = circuit_sequence
                circuit_sequence += 1
                # appends a tuple to ensure all the data is kept on the circuit
                euler_circuit.append((edge[0], edge[1], edge_att))
                continue

            # edge is augmented, find the shortest 'real' route
            # since augmented paths are just straight lines between the points, we need to determine the path along the
            # original graph to get the real route
            aug_path = nx.shortest_path(graph_original, edge[0], edge[1], weight=self.weight_field)
            aug_path_pairs = list(zip(aug_path[:-1], aug_path[1:]))

            # add the edges from the shortest path to the final circuit
            for edge_aug in aug_path_pairs:
                edge_aug_att = graph_original.get_edge_data(edge_aug[0], edge_aug[1])[0]
                edge_aug_att[self.sequence_field_name] = circuit_sequence
                circuit_sequence += 1
                euler_circuit.append((edge_aug[0], edge_aug[1], edge_aug_att))

        return euler_circuit

class EdgeOrder:
    """Given a graph and a start point, label each edge with an order in which it should be enumerated if walking every
    edge in the graph."""

    def __init__(self, graph, seq_start=1):

        logging.debug("Initializing edge order on graph")
        self.graph = graph
        self.sequence = seq_start

        self.missed_edges = collections.deque()
        self.labels = {}

        if not nx.is_connected(self.graph):
            logging.info("Graph is disconnected. Edge order should be verified.")


    def _sort_edges_by_count(self, start, ends):
        """Sort the edges connected to a node based on the number of edges connected to each node in ends.

        This produces a list of edges connected to the start node, sorted from highest edge count to lowest.
        """

        logging.debug("Counting edges connected to %s", start)

        edge_counts = {}
        for end in ends:
            edge_counts[(start, end)] = self.graph.number_of_edges(start, end)
        sorted_edges = sorted(edge_counts.items(), key=lambda kv: kv[1], reverse=True)
        logging.debug("Edge counts: %s", sorted_edges)
        return sorted_edges


    def _node_intersects_missed_edge(self, node):
        """Check if the given node intersects any of the edges that were marked as missed."""

        logging.debug("Looking for %s in missed edges", node)
        # look for the node in the missed edges list
        for ed in self.missed_edges:
            if (node in ed):
                logging.debug("Found %s as intersecting edge", ed)
                return ed

        logging.debug("No intersecting edge found.")
        return None


    def _apply_sequence_to_edge(self, u, v, k):
        """Generate a label for the given edge, applying the current sequence value and increasing the sequence
        counter by 1.
        """


        edge_label = (u, v, k)
        self.labels[edge_label] = self.sequence
        logging.debug("Labelling {} as {}".format(edge_label, self.sequence))

        self.sequence += 1


    def _apply_sequence_to_edges(self, u, v):
        """Iterate all the edges between the start and end node, labelling them with a sequence value based on their
        order of appearance within the graph.

        Labels that already exist from the graph as skipped without increasing the sequence counter.
        """

        edge_count = self.graph.number_of_edges(u, v)
        logging.debug("Applying sequence to %s edges from %s to %s", edge_count, u, v)
        for k in range(edge_count):
            edge_label = (u, v, k)

            # skip anything that's already been seen
            if edge_label in self.labels:
                logging.debug("%s already labelled, skipping", edge_label)
                continue

            self._apply_sequence_to_edge(u, v, k)


    def _get_start_node_for_first_edge(self, graph, key='sequence'):
        """Find the first edge and return the start node for it."""

        logging.debug("Looking for start node based on %s", key)
        attribs = nx.get_edge_attributes(graph, key)
        start_node = None
        seg_num = -1
        for edge, val in attribs.items():
            # sequence numbers don't always start at 0, so need to find the lowest number and go from there.
            if val < seg_num or seg_num == -1:
                seg_num = val
                start_node = edge[0]

        # return the start node for the edge
        logging.debug("Found %s as the starting node with %s = %s", start_node, key, seg_num)
        return start_node


    def label_edges(self):
        """Label all the edges in the graph with a sequence number, starting from the given first edge.

        This results is a dictionary of labels that can then be applied to the graph. It does not set the labels on the
        graph itself.

        Graphs can be disconnected, meaning not all edges have an obvious way between them. This is handled by working
        through the connected components of the graph from the largest to the smallest. No consideration is given to
        this problem, and the sequence number just keeps incrementing. This means some components can be spatially
        far apart and have seemingly non-sensical sequence values.
        """

        # work on each connected component of the graph from largest to smallest
        # for fully connected graphs, there will only be one component
        for comp in sorted(nx.connected_components(self.graph), key=len, reverse=True):
            graph_component = self.graph.subgraph(comp)
            start_node = self._get_start_node_for_first_edge(graph_component)

            # find all the successors from the start node using a depth first search
            logging.debug("Searching for successors from %s", start_node)
            successors = nx.dfs_successors(graph_component, start_node)
            logging.debug("Successors: %s", successors)

            logging.debug("Walking the edges in the graph")
            for node in successors:
                start = node
                ends = successors[node]
                logging.debug("Traversing from node %s", start)

                edge_counts = self._sort_edges_by_count(start, ends)

                # process each edge
                logging.debug("Iterating each edge")
                for edge_nodes, count in edge_counts:
                    logging.debug("Processing edges between %s", edge_nodes)
                    u, v = edge_nodes

                    # if the end node has children, and there are two edges, mark this
                    # to be processed later
                    if (v in successors) and (count > 1):
                        # only do one edge and move down stream
                        self._apply_sequence_to_edge(u, v, 0)

                        logging.debug("%s has children and multiple edges. Marking to come back later.", v)
                        self.missed_edges.append(edge_nodes)

                        logging.debug("Moving to next node in successors")
                        break

                    # sequence all the edges we are on right now (both sides of the road
                    logging.debug("No successors found, labelling all edges between %s", edge_nodes)
                    self._apply_sequence_to_edges(u, v)

                    # see if all the successor nodes have been consumed before moving on to missed edges
                    logging.debug("Looking for more work to do before evaluating missed edges")
                    more_nodes_flag = False
                    for successor in ends:
                        successor_edge = (start, successor, 0)
                        if successor_edge not in self.labels:
                            logging.debug("Found more successors to process from %s", u)
                            # set the flag and move on to the next nodes
                            more_nodes_flag = True
                            break
                    # avoid checking for missed nodes if there was more to do here
                    if more_nodes_flag:
                        logging.debug("More edges to be processed before carrying on")
                        continue
                    else:
                        logging.debug("No more unseen edges to process.")

                    # check if there are missed edges to backtrack over
                    logging.debug("Looking for any previously missed edges that may intersect %s", u)
                    missed_edge = self._node_intersects_missed_edge(u)
                    if missed_edge:
                        logging.debug("Found missed edge %s. Applying sequence label.", missed_edge)
                        self._apply_sequence_to_edges(missed_edge[0], missed_edge[1])
                        logging.debug("Marking missed edge as complete")
                        self.missed_edges.remove(missed_edge)

            # return edges that connect to the original start point won't be capture
            # in the check for successors, so do them here
            logging.debug("Checking for missed return edge from %s", start_node)
            return_edges = graph_component.edges(nbunch=start_node)
            logging.debug("Evaluating %s edges as possible return path", len(return_edges))
            # look for any that aren't labeled
            for re in return_edges:
                edge_count = graph_component.number_of_edges(re[0], re[1])
                for edge_num in range(edge_count):
                    edge_id = (re[0], re[1], edge_num)
                    # skip any that have already been seen
                    if edge_id in self.labels:
                        logging.debug("%s is already labelled, so not a return edge", edge_id)
                        continue
                    # set the sequence on missing connections
                    self._apply_sequence_to_edge(re[0], re[1], edge_num)

            # if any edges were missed, assign them a sequence value
            # this is not a desired state and would ideally never need to happen
            for edge in graph_component.edges:
                # skip anything that has been seen
                if edge in self.labels:
                    continue
                logging.warning("Applying out of order sequence to %s", edge)
                self._apply_sequence_to_edge(edge[0], edge[1], edge[2])

        # return all the labels with the sequence counts applied
        return self.labels
