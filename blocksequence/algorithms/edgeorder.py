"""
Ordering of edges within a block using DiGraphs as much as possible.
"""


from collections import Counter
import logging
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ['EdgeOrder']

class EdgeOrder:
    """Sequence the edges within a given parent geography.

    This will generate a Eulerian path for all the edges in each child geography found within the parent geography.
    When no possible path can be found, the edges are augmented to create a circuit that is routable. These edges are
    then (optionally) removed from the final results. When this happens the enumerator will have to backtrack to
    traverse on the road arcs that form the edges in a block graph.
    """

    def __init__(self, pgeo_edges, source='source', target='target', weight='length', anomalies_path=None):
        """Initialize the sequencer with a DataFrame representing the edges in a parent geography.

        This will create a MultiGraph from the input edges that can be sequenced later.

        Parameters
        ----------
        pgeo_edges : Pandas DataFrame
            The edges that exist within a single parent geography.

        source : String, 'source'
            The name of the column in pgeo_edges to be used as the source node identifier.

        target : String, 'target'
            The name of the column in pgeo_edges to be used as the target node identifier.

        weight : String, 'length'
            The name of the column in pgeo_edges to be used as a weight value for each edge.

        anomalies_path : pathlib.Path
            The path to a folder where anomalous edges will be output for future analysis
        """

        logger.debug("EdgeSequence class initialization started")

        self.edges = pgeo_edges
        self.source_field = source
        self.target_field = target
        self.weight_field = weight

        self.anomaly_folder = anomalies_path

        logger.debug("Received %s edges", len(self.edges))
        logger.debug("Source column: %s", self.source_field)
        logger.debug("Target column: %s", self.target_field)
        logger.debug("Weight field: %s", self.weight_field)

        # the type of graph to use when processing the data
        self.graph_type = nx.MultiDiGraph

        # field names to use on the outputs
        self.eo_name = 'edge_order'

        # names for attributes used to track internal workings
        self.node_weight_attr = 'weight'
        self.node_interior_edge_attr = 'interior'

        # initialize the graph
        self.graph = self._create_graph()
        if nx.is_empty(self.graph):
            logger.error("Empty graph created. Nothing to sequence.")
        logger.debug("Initialized graph: %s", self.graph.edges(data=True))

        logger.debug("EdgeSequence initialized")

    def _create_graph(self):
        """
        Build a networkx graph from the edge list.

        This takes the pandas DataFrame that was supplied and converts it to a graph that is then stored on in the
        instance for use by other methods.
        """

        logger.debug("Converting pandas edge list to graph")
        return nx.from_pandas_edgelist(self.edges, self.source_field, self.target_field, edge_attr=True,
                                       create_using=self.graph_type)

    def sequence(self, cgeo_attr, drop_augmented=True):
        """Produce a graph that includes sequence values for the blocks within the parent geography, as well as the
        edge ordering within those blocks.

        Parameters
        ----------
        cgeo_attr : String
            The column of the child geography UID column within the DataFrame that was provide when the class was
            instantiated.

        drop_augmented : bool, True
            Whether or not to drop any edges that needed to be added to a block in order to find a suitable route.

        Returns
        -------
        edges : pandas.DataFrame
            The DataFrame that was provided during instantiation with a column that represents the order of the edges
            in each block.
        """

        logger.debug("sequence started")

        # Set the initial weight of each node to be the degree of the node within the parent geometry.
        # This will be used to find the best start points for each block.
        logger.debug("Initializing node weights based on degree values")
        nx.set_node_attributes(self.graph, dict(self.graph.degree()), name=self.node_weight_attr)
        logger.debug("Node weights at start: %s", nx.get_node_attributes(self.graph, self.node_weight_attr))

        # Identify nodes on interior arcs so that these aren't preferred starting points.
        nodes_on_interior_edges = set()
        logger.debug("Interior edges: %s", nx.get_edge_attributes(self.graph, self.node_interior_edge_attr))
        for edge, flag in nx.get_edge_attributes(self.graph, self.node_interior_edge_attr).items():
            u, v, k = edge
            if flag == 1:
                nodes_on_interior_edges.add(u)
                nodes_on_interior_edges.add(v)
        node_labels = dict(zip(nodes_on_interior_edges, [True] * len(nodes_on_interior_edges)))
        logger.debug("Flagging %s nodes as being on interior edges", len(node_labels))
        nx.set_node_attributes(self.graph, node_labels, name=self.node_interior_edge_attr)
        logger.debug("Nodes on interior edges: %s", nodes_on_interior_edges)
        # scale the interior nodes down in the weighting
        self._apply_node_scaling_factor(nodes_on_interior_edges, factor=-.2)

        # Get the block IDs by chosen order.
        logger.debug("Processing edges based on the chosen block order")
        block_order = self.edges.sort_values('block_order')[cgeo_attr].unique()
        logger.debug("Block order: %s", block_order)

        # Group the edge listing by the child geography (blocks) so that each can be enumerated independently.
        block_groups = self.edges.groupby(cgeo_attr, sort=False)
        logger.debug("Found %s blocks by %s", len(block_groups), cgeo_attr)
        results = []
        for block_id in block_order:
            block_edges = block_groups.get_group(block_id)

            logger.debug("Processing block %s", block_id)
            seq_edges = self._order_edges_in_block(block_edges, drop_augmented)

            # Check that the results of edge ordering align with the number of total edges in the block.
            if len(block_edges) < seq_edges['edge_order'].max():
                logging.critical("Edge order (%s) exceeds the number of edges (%s) in block %s",
                                 seq_edges['edge_order'].max(), len(block_edges), block_id)
            elif len(block_edges) > seq_edges['edge_order'].max():
                logging.critical("Edge order (%s) is less than the number of edges (%s) in block %s",
                                 seq_edges['edge_order'].max(), len(block_edges), block_id)
            else:
                logging.debug("Edge count matches order values in block %s", block_id)

            if seq_edges[self.eo_name].isnull().any():
                logger.critical("Null values found in block %s", block_id)

            # Store the results until the entire parent geography is complete.
            results.append(seq_edges)

        # Merge all the results into a single dataframe and put the resulting order onto the initially provided data.
        logger.debug("Merging edge order onto supplied edge list")
        eo_df = pd.concat(results)
        self.edges = self.edges.merge(eo_df[['bf_uid', self.eo_name, 'path_type']], how='left', on='bf_uid')

        # Sanity check that the results aren't null
        if self.edges[self.eo_name].isnull().any():
            logger.critical("Null edge order found within blocks: %s.", self.edges[cgeo_attr].unique().tolist())

        return self.edges

    def _order_edges_in_block(self, block_data, drop_augmented):
        """Produce an edge sequence for all edges in the component.

        Parameters
        ----------
        block_data : pandas.DataFrame
            A DataFrame representing all the edges within a single block.

        drop_augmented : bool
            Whether or not to keep any edges that needed to be added to the source edges in order to navigate the
            network.

        Returns
        -------
        edges : pandas.DataFrame
            The same edges that were input with the edge order and route type as new columns.
        """

        logger.debug("order_edges_by_block started")

        logger.debug("Received edge data of shape %s", block_data.shape)
        # Sort the DataFrame to load right hand arcs into NetworkX first.
        # Note that Eulerian paths work in reverse order.
        block_data = block_data.sort_values('arc_side', ascending=False)
        block_g = nx.from_pandas_edgelist(block_data, source=self.source_field, target=self.target_field,
                                          edge_attr=True, create_using=self.graph_type)

        logger.debug("Block contains %s edges and %s nodes", block_g.number_of_edges(), block_g.number_of_nodes())

        # if the graph is empty it means there is a problem with the source data
        # an error is logged, but other blocks are still processed
        if nx.is_empty(block_g):
            logger.error("Block contains no edges and cannot be sequenced")
            return

        # Scale nodes that are mid-segment by looking for duplicated ngd_str_uid values
        logger.debug("Looking for nodes that fall in the middle of a road segment")
        block_data['same_ngd_str_uid'] = block_data.duplicated(subset=['ngd_str_uid'], keep=False)
        mid_arc_start_nodes = set(block_data.loc[block_data['same_ngd_str_uid'] == True, 'startnodenum'])
        mid_arc_end_nodes = set(block_data.loc[block_data['same_ngd_str_uid'] == True, 'endnodenum'])
        mid_arc_nodes = mid_arc_start_nodes.intersection(mid_arc_end_nodes)
        if mid_arc_nodes:
            logger.debug("Found mid-segment nodes: %s", mid_arc_nodes)
            self._apply_node_scaling_factor(mid_arc_nodes, factor=-0.5)

        # initialize the edge sequence counter
        edge_sequence = 0

        # record what type of path was used to determine the circuit
        path_indicator_name = 'path_type'
        path_indicator_edges = {}

        # blocks don't necessarily form fully connected graphs, so cycle through the components
        logger.debug("Block contains %s connected components", nx.number_weakly_connected_components(block_g))
        for block_comp in sorted(nx.weakly_connected_components(block_g), key=len, reverse=True):
            logger.debug("Creating subgraph from connected component with %s nodes", len(block_comp))
            block_g_comp = block_g.subgraph(block_comp)

            # determine the preferred start node for this block component
            preferred_sp = self._get_preferred_start_node(block_g_comp.nodes)
            logger.debug("Preferred start node for this block: %s", preferred_sp)

            logger.debug("Component contains %s edges and %s nodes", block_g_comp.number_of_edges(), len(block_g_comp))

            # Need to pick an approach to processing this component depending on what type of circuit it forms.
            # Ideally things are a Eulerian circuit that can be walked and return to start, but not all blocks form
            # these nice circuits. If no good circuit can be found, then sequence numbers are just applied but may
            # not form a logical order.

            # Track the sequence value in case the enumeration method needs to be reset. This gets used when using
            # the preferred start point fails, and also controls if the start node for this component is marked as a
            # point we want to cluster on.
            seq_val_at_start = edge_sequence

            # the preferred option is a Eulerian circuit, so try that first
            # logger.debug("Available edges: %s", block_g_comp.edges)
            if nx.is_eulerian(block_g_comp):
                logger.debug("Block component is eulerian.")
                # record all these edges as being eulerian
                indicator = dict(zip(block_g_comp.edges, ['circuit'] * block_g_comp.size()))
                path_indicator_edges.update(indicator)

                # enumerate the edges and order them directly
                logger.debug("Creating Eulerian circuit from node %s", preferred_sp)
                for u, v, k in nx.eulerian_circuit(block_g_comp, source=preferred_sp, keys=True):
                    edge_sequence += 1
                    block_g.edges[u, v, k][self.eo_name] = edge_sequence
                    # logger.debug("Sequence applied: (%s, %s, %s) = %s", u, v, k, edge_sequence)

            # next best option is a path that stops at a different location from the start point
            elif nx.has_eulerian_path(block_g_comp):
                logger.debug("Block component forms Eulerian path")

                # record all these edges as being a eulerian path
                indicator = dict(zip(block_g_comp.edges, ['path'] * block_g_comp.size()))
                path_indicator_edges.update(indicator)

                try:
                    logger.debug("Trying to create path from preferred start node %s", preferred_sp)
                    for u, v, k in nx.eulerian_path(block_g_comp, source=preferred_sp, keys=True):
                        edge_sequence += 1

                        # check if the start point is actually in the first edge
                        if edge_sequence == 1 and not (preferred_sp == u or preferred_sp == v):
                            logger.debug("Preferred start point not present on starting edge, throwing KeyError.")
                            raise KeyError("Invalid starting edge")

                        # Sometimes the preferred start point means walking over the same edge twice, which will leave
                        # a data gap (the previous edge order value will be overwritten). If this happens, throw a
                        # KeyError
                        if block_g.edges[u, v, k].get(self.eo_name):
                            logger.debug("Edge already sequenced.")
                            raise KeyError("Preferred start point results in backtracking.")

                        block_g.edges[u, v, k][self.eo_name] = edge_sequence
                        # logger.debug("Sequence applied: (%s, %s, %s) = %s", u, v, k, edge_sequence)

                        if edge_sequence < block_g_comp.number_of_edges():
                            logger.debug("It looks like some edges got missed")
                            raise KeyError("Missing edges on path")

                    logger.debug("Path was created from desired start point %s", preferred_sp)
                except KeyError:
                    # preferred start point failed; let networkx pick and start over
                    logger.debug("Preferred start node did not create a path. Trying a different one.")

                    # reset the path listing since a new point will be picked
                    logger.debug("Reset edge_sequence value to %s", seq_val_at_start)
                    edge_sequence = seq_val_at_start

                    for u, v, k in nx.eulerian_path(block_g_comp, keys=True):
                        edge_sequence += 1
                        block_g.edges[u, v, k][self.eo_name] = edge_sequence
                        # logger.debug("Sequence applied: (%s, %s, %s) = %s", u, v, k, edge_sequence)

            # No good path exists, which means someone will have to backtrack
            else:
                logger.debug("Non-eulerian block is not easily traversable. Eulerizing it.")

                # Record all these edges as being augmented.
                indicator = dict(zip(block_g_comp.edges, ['augmented'] * block_g_comp.size()))
                path_indicator_edges.update(indicator)

                # Send this data to the anomaly folder so that it can be investigated later. It could have addressable
                # issues that operations can correct for the next run.
                logger.debug("Writing anomaly set for this block")
                bf_uid_set = list(nx.get_edge_attributes(block_g_comp, 'bf_uid').values()).pop()
                anomaly_file_name = f"anomaly_block_component.{bf_uid_set}.yaml"
                nx.write_yaml(block_g_comp, (self.anomaly_folder / anomaly_file_name).as_posix())

                # You cannot eulerize a directed graph, so create an undirected one
                logger.debug("Creating MultiGraph from directed graph.")
                temp_graph = nx.MultiGraph()
                for u, v, data in block_g_comp.edges(data=True):
                    key = temp_graph.add_edge(u, v, **data)
                    # logger.debug("Adding edge (%s, %s, %s) to temporary graph.", u, v, key)
                logger.debug("Created temporary MultiGraph with %s edges", temp_graph.number_of_edges())

                # Convert the temporary graph to a proper Euler circuit so that it can be traversed.
                logger.debug("Eulerizing MultiGraph")
                euler_block = nx.eulerize(temp_graph)
                logger.debug("Added %s edges to the block", (euler_block.size() - temp_graph.size()))
                logger.debug("Number of vertices in eulerized graph: %s", euler_block.number_of_nodes())

                # As we try to traverse the undirected graph, we need to keep track of places already visited to make
                # sure arcs are not skipped.
                visited_edges = Counter()

                # augmented edges will throw the node weights off, so don't bother trying the preferred start node
                logger.debug("Generating path through augmented block")
                for u, v, k in nx.eulerian_circuit(euler_block, preferred_sp, keys=True):
                    # augmented edges have no attributes, so look for one and skip the edge if nothing is returned
                    if drop_augmented and not euler_block.edges[u, v, k].get('bf_uid'):
                        logger.debug("Ignoring augmented edge (%s, %s, %s)", u, v, k)
                        continue

                    # Increment the sequence value for each edge we see.
                    edge_sequence += 1

                    # Since we formed an undirected MultiGraph we need to check the orientation of the nodes on the
                    # edge to assign the sequence back to the directed graph.
                    start_node = u
                    end_node = v
                    available_edge_count = block_g.number_of_edges(start_node, end_node)
                    # If no edges exist, invert the nodes and check again.
                    # This also checks to see if we've already encountered all the edges between these nodes, indicating
                    # we need to process the inverse of the start and end values
                    if available_edge_count == 0 or (
                            ((start_node, end_node) in visited_edges) and
                            (available_edge_count == visited_edges[(start_node, end_node)])):
                        logger.debug("Nothing to process between (%s, %s), inverting nodes.", start_node, end_node)
                        start_node = v
                        end_node = u
                        available_edge_count = block_g.number_of_edges(start_node, end_node)
                    logger.debug("Number of edges available between (%s, %s): %s", start_node, end_node,
                                 available_edge_count)

                    # Apply the edge_sequence to the first edge that hasn't received one yet
                    for ki in range(available_edge_count):
                        if not block_g.edges[start_node, end_node, ki].get(self.eo_name):
                            logger.debug("Edge sequence applied: (%s, %s, %s) = %s", start_node, end_node, ki,
                                         edge_sequence)
                            block_g.edges[start_node, end_node, ki][self.eo_name] = edge_sequence
                            visited_edges[(start_node, end_node)] += 1
                            break

            # At this point every edge should be accounted for, but in case something somehow slips through the cracks
            # it needs to be given a sequence label. The label almost certainly won't make much sense in terms of a
            # logical ordering, but this is just trying ot make sure it is counted.
            logger.debug("Looking for any missed edges in block component")
            for u, v, k in block_g_comp.edges:
                if not block_g.edges[u, v, k].get(self.eo_name):
                    edge_sequence += 1
                    block_g.edges[u, v, k][self.eo_name] = edge_sequence
                    logger.warning("Applied out of order sequence to component edge (%s, %s, %s): %s", u, v, k,
                                   edge_sequence)

            # just log the last sequence value to make tracing easier
            logger.debug("Final edge sequence value for component: %s", edge_sequence)

            # apply a sequence value to all the edges that were discovered
            logger.debug("Edge order results: %s", nx.get_edge_attributes(block_g_comp, self.eo_name))

            # To help cluster the start nodes, mark which node was used as the start point in this block
            if seq_val_at_start == 1:
                self._mark_chosen_start_node(block_g_comp, preferred_sp)

            logger.debug("Finished processing component")

        # record that block processing is finished
        logger.debug("Block processing complete")

        # nx.set_edge_attributes(block_g, block_sequence_labels, self.eo_name)
        nx.set_edge_attributes(block_g, path_indicator_edges, path_indicator_name)

        # check to see if the counts line up
        if not block_g.number_of_edges() == edge_sequence:
            logger.debug("Edge sequence (%s) and edge count (%s) do not match in block",
                         edge_sequence, block_g.number_of_edges())

        # help start point clustering by apply a scaling factor to all nodes that were touched
        logger.debug("Applying scaling factor to nodes in this block, except start point")
        nodes_in_block = set(block_g.nodes())
        nodes_in_block.remove(preferred_sp)  # don't scale the preferred start point
        self._apply_node_scaling_factor(nodes_in_block)
        logger.debug("Final node data for block: %s", self.graph.subgraph(block_g.nodes).nodes(data=True))

        logger.debug("Returning pandas DataFrame from block graph.")
        return nx.to_pandas_edgelist(block_g, source=self.source_field, target=self.target_field)

    def _mark_chosen_start_node(self, graph, preferred_sp):
        """Flag the start point used for building the cycle so that other blocks can find it.

        Parameters
        ----------
        graph : Graph
            The graph that was enumerated

        preferred_sp : node
            The start node that was registered as being preferred
        """

        logger.debug("_mark_chosen_start_node started")

        # best case is that the preferred node was used
        chosen_sp = preferred_sp

        # Validate that the preferred node actually generated the sequence.
        edges_on_pref_sp = graph.edges(nbunch=preferred_sp, data=True)

        # Cycle through the edges on the preferred point to see if the edge is sequence 1.
        for edge in edges_on_pref_sp:
            # only care about the edge marked 1 and if the preferred node is not on it
            if (edge[2][self.eo_name] == 1) and (preferred_sp not in edge):
                logger.debug("Preferred start node not used on this graph")
                chosen_sp = None

        # If the preferred node wasn't used, go find it.
        if not chosen_sp:
            logger.debug("Finding start node on first edge in block")
            # This isn't perfect, as it will use the source node on the edge. It is possible the target node was the
            # actual start point, but there is no way to know.
            chosen_sp = sorted(nx.get_edge_attributes(graph, self.eo_name).items(), key=lambda x: x[1])[0][0][0]

        # Set the start point flag on the node
        logger.debug("Recording node %s as start point", chosen_sp)
        self.graph.nodes[chosen_sp]['sp'] = True

    def _apply_node_scaling_factor(self, node_set, factor=-0.1):
        """Apply a scaling factor to all nodes in a given graph except the starting point provided.

        This alters the value of the weight attribute in self.graph directly.

        Parameters
        ----------
        node_set : set
            The nodes within the graph that will have their weights scaled.

        factor : float
            The scaling factor to be applied to the nodes in node_set.
        """

        logger.debug("Applying scaling factor of %s to nodes", factor)

        # get all the nodes in the block
        node_weights = dict(nx.get_node_attributes(self.graph, self.node_weight_attr))

        # reduce the weights slightly for those on interior arcs
        for node in node_set:
            if node in node_weights:
                node_weights[node] += factor

        nx.set_node_attributes(self.graph, node_weights, self.node_weight_attr)

        logger.debug("New node weights: %s", self.graph.subgraph(node_set).nodes(data=True))

    def _get_preferred_start_node(self, nodes):
        """Pick a preferred start node for traversing a block based on the node weights.

        The nodes with the highest weight are picked first, but if the node is on an interior arc it is scaled
        slightly to try and minimize the number of times an internal arc is picked as a preferred starting position.

        Parameters
        ----------
        nodes : set
            The set of nodes available to be used as a start point.

        Returns
        -------
        optimal_sp : node
            The preferred node to be used as a starting position.
        """

        logger.debug("get_preferred_start_node started")

        # get access to the node info for the chosen nodes
        graph = self.graph.subgraph(nodes)
        logger.debug("Graph nodes: %s", graph.nodes(data=True))

        # Get the node weights for all the nodes available on this subgraph
        node_weights = dict(nx.get_node_attributes(graph, self.node_weight_attr))
        logger.debug("Initial node weights: %s", node_weights)

        # Nodes that were previously used as start points get a boost in weighting.
        prev_sp_nodes = dict(nx.get_node_attributes(graph, 'sp'))
        logger.debug("Previous start point nodes: %s", prev_sp_nodes)
        for previous_sp in prev_sp_nodes:
            node_weights[previous_sp] += 10

        logger.debug("Potential start nodes by weight: %s", node_weights)
        # Order the nodes by their calculated weights.
        ordered_start_nodes = sorted(node_weights.keys(), key=lambda w: node_weights[w], reverse=True)
        logger.debug("Preferred start node order in this node set: %s", ordered_start_nodes)
        # The optimal node is the one with the highest weighting.
        optimal_sp = ordered_start_nodes[0]
        logger.debug("Optimal start point chosen: %s, weight: %s", optimal_sp, node_weights[optimal_sp])

        return optimal_sp
