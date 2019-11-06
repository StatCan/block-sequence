import itertools

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
