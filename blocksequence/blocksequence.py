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
        start_nodes = self._get_preferred_start_nodes(graph)

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

        # calculate the block order
        df['block_order'] = block_group.ngroup() + 1

        edge_order_field = 'edge_order'
        eo_seq = []
        for block, group in block_group:
            edge_graph = nx.from_pandas_edgelist(group,
                                                 source='startnodenum', target='endnodenum',
                                                 edge_attr=True, create_using=nx.MultiGraph)
            eo = EdgeOrder(edge_graph)
            edge_labels = eo.label_all_edges()
            nx.set_edge_attributes(edge_graph, edge_labels, edge_order_field)

            # dump it back to pandas to be joined to the rest of the data
            edge_df = nx.to_pandas_edgelist(edge_graph)[['bf_uid', edge_order_field]]
            eo_seq.append(edge_df)

        # put the edge_order values onto the edges from this geography
        eo_df = pd.concat(eo_seq)
        df = df.merge(eo_df, on='bf_uid')

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

    def _get_preferred_start_nodes(self, graph):
        """Generate a list of start points to be used in route evaluations."""

        # boundary arcs are those with only one edge, where interior arcs have two edges.
        boundary_edges = [e for e in graph.edges() if graph.number_of_edges(e[0], e[1]) == 1]
        nodes_on_boundary = set()
        for u, v in boundary_edges:
            nodes_on_boundary.add(u)
            nodes_on_boundary.add(v)

        # sort the nodes by popularity
        node_list = sorted(dict(graph.degree(nbunch=nodes_on_boundary)).items(), key=lambda kv: kv[1], reverse=True)
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

        self._es_label = 'es'

        if not nx.is_connected(self.graph):
            logging.info("Graph is disconnected. Edge order should be verified.")


    def _sort_edges_by_count(self, start: int, ends: list) -> list:
        """Sort the edges connected to a node based on the number of edges connected to each node in ends.

        This produces a list of edges connected to the start node, sorted from highest edge count to lowest.
        """

        logging.debug("Counting edges connected to %s from set %s", start, ends)

        edge_counts = {}
        for end in ends:
            edge_counts[(start, end)] = self.graph.number_of_edges(start, end)
        sorted_edges = sorted(edge_counts.items(), key=lambda kv: kv[1], reverse=True)
        logging.debug("Edge counts: %s", sorted_edges)
        return sorted_edges


    def _apply_sequence_to_edge(self, u, v, k):
        """Generate a label for the given edge, applying the current sequence value and increasing the sequence
        counter by 1.
        """

        # guard against resetting an already set label
        if self.graph.edges[u, v, k].get(self._es_label):
            return

        self.graph.edges[u, v, k][self._es_label] = self.sequence
        logging.debug("Labelling {} as {}".format((u,v,k), self.sequence))

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
            if edge_label in nx.get_edge_attributes(self.graph, self._es_label):
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

    def _phantom_successors(self, graph, node, known_ends):
        """Look for edges that form connections between nodes which aren't represented in successors."""

        logging.debug("Looking for connections not listed in successors to %s.", node)

        edges_from_node = graph.edges(nbunch=node)
        connected_nodes = set([u for u,v in edges_from_node])

        unseen_connections = connected_nodes - set(known_ends)
        logging.debug("Found %s phantom successors: %s", len(unseen_connections), unseen_connections)

        logging.debug("Final %s phantom successors: %s", len(unseen_connections), unseen_connections)
        return unseen_connections


    def _find_start_path(self, G):
        """Return a suitable starting vertex for an Eulerian path.

        This is a backport of the method in networkx 2.4. Only undirected MultiGraphs as supported by this backport.
        """

        if not self._has_eulerian_path(G):
            return None

        if nx.is_eulerian(G):
            return nx.utils.arbitrary_element(G)

        # randomly choose one of the possibilities
        start = [v for v in G if G.degree(v) % 2 != 0][0]
        return start

    def _is_semieulerian(self, G):
        """Return True if a graph is semi-Eulerian.

        A graph is semi-Eulerian if it has an Eulerian path but no Eulerian circuit.

        This is a backport of the method in networkx 2.4, since we don't have that version at the moment.
        """

        return self._has_eulerian_path(G) and not nx.is_eulerian(G)


    def _has_eulerian_path(self, G):
        """Return True if G has an Eulerian path.

        An Eulerian path is a path in a graph which uses each edge of a graph exactly once.

        This is a backport of a method in networkx 2.4, and is only implemented for undirected graphs in this port.

        Parameters
        ----------
        G: Networkx undirected Graph
           The graph to find an euler path in.

        Returns
        -------
        Bool: True if G has an eulerian path.
        """

        return (sum(d % 2 == 1 for v, d in G.degree()) in (0, 2) and nx.is_connected(G))


    def _eulerian_path(self, G, source=None):
        """Return an iterator over the edges of an Eulerian path in G.

        This is a backport of a method in networkx 2.4, and is only implemented for undirected MultiGraphs in this port.

        Parameters
        ----------
        G: NetworkX graph
           The graph in which to look for an eulerian path.
        source: node
            The node at which to start the search.

        Yields
        ------
        Edge tuples along the eulerian path.
        """

        # protect against graphs with no possible path
        if not self._has_eulerian_path(G):
            raise nx.NetworkXError("Graph has no Eulerian paths.")

        # make a copy of the graph
        GC = G.copy()

        if source is None:
            source = self._find_start_path(GC)

        # traverse the edges using the networkx internal multigraph eulerian process
        for u, v, k in nx.algorithms.euler._multigraph_eulerian_circuit(GC, source):
            yield u, v, k


    def label_all_edges(self):
        logging.info("Labelling all edges recursively.")
        for comp in sorted(nx.connected_components(self.graph), key=len, reverse=True):
            graph_component = self.graph.subgraph(comp)
            start_node = self._get_start_node_for_first_edge(graph_component)

            # try to route a eulerian circuit as the simplest solution before trying anything else
            if nx.is_eulerian(graph_component):
                logging.debug("Graph is eulerian. Labelling as eulerian circuit.")
                for u,v,k in nx.eulerian_circuit(graph_component, source=start_node, keys=True):
                    self._apply_sequence_to_edge(u,v,k)
                continue

            # check if a eulerian path can be produced
            if self._has_eulerian_path(graph_component):
                logging.debug("Component has eulerian path. Labelling with that.")
                try:
                    for u,v,k in self._eulerian_path(graph_component, start_node):
                        self._apply_sequence_to_edge(u, v, k)
                except:
                    # sometimes picking the start point ourselves doesn't end up in a spot that can produce a good path,
                    # so we let it pick its own.
                    for u,v,k in self._eulerian_path(graph_component):
                        self._apply_sequence_to_edge(u, v, k)

                # component is fully processed. Move on to next one
                continue

            # no eulerian path exists, so ask networkx to create one by filling in edges where necessary
            # the new edges will need to be removed from the final output, but should let the system walk through an
            # area
            logging.debug("Non-eulerian graph is not traversable. Eulerizing it.")
            euler_graph = nx.eulerize(graph_component)
            logging.debug("New graph: %s", euler_graph.edges)
            for u,v,k in self._eulerian_path(euler_graph):
                # make sure this is a real block face before trying to apply a sequence value to it
                if not euler_graph[u][v][k].get('bf_uid'):
                    logging.debug("Augmented edge at (%s, %s, %s) found, not applying sequence number.", u, v, k)
                    continue

                self._apply_sequence_to_edge(u, v, k)

            # At this point every edge should be accounted for, but in case something somehow slips through the cracks
            # it needs to be given a sequence label. The label almost certainly won't make much sense in terms of a
            # logical ordering, but this is just trying to make sure it is counted.
            for u, v, k in graph_component.edges:
                if not graph_component[u][v][k].get(self._es_label):
                    self._apply_sequence_to_edge(u, v, k)

        # collect all the labels that were applied and return them
        labels = nx.get_edge_attributes(self.graph, self._es_label)
        logging.debug("Final labels: %s", labels)
        return labels

    def _label_from_node(self, node, successors):
        """Label all the edges from a given node."""

        ends = successors[node]
        logging.debug("Recursive Traversing from node %s", node)

        edge_counts = self._sort_edges_by_count(node, ends)

        # process each edge
        for edge_nodes, count in edge_counts:
            u, v = edge_nodes

            # if the end node has children, and there are two edges, label this and
            # move on before doing the rest
            if (v in successors) and (count > 1):
                self._apply_sequence_to_edge(u,v,0)

                # start again from the end point
                self._label_from_node(v, successors)

            self._apply_sequence_to_edges(u, v)
