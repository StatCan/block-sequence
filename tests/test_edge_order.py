import logging
import networkx as nx
import pprint

from blocksequence.blocksequence import EdgeOrder

def path_works(graph, labels):
    # try to walk the graph using the labels
    sorted_labels = sorted(labels, key=lambda t: t[1])
    logging.debug(sorted_labels)
    for label in sorted_labels:
        neighbors = [i for i in nx.neighbors(graph, label[0])]
        if label[1] not in neighbors:
            return False
    
    # the highest label number should match the number of edges
    edge_count = len(graph.edges)
    highest_label = max(labels.items(), key=lambda t: t[1])[1]
    if edge_count != highest_label:
        return False
    
    # the number of labels should match the number of edges
    if edge_count != len(labels):
        return False

    return True


def test_basic_cycle(caplog):
    """Test the labelling of edges in a small cycle.

    This tests that the order will follow a basic cycle when the block arcs all connect in a simple circle.

    """

    # capture debug logs on failures
    caplog.set_level(logging.DEBUG)

    # create a small graph and label the edges
    g = nx.cycle_graph(5, create_using=nx.MultiGraph)
    # edge order is going to look for a sequence field to determine the start node
    seq = {(0, 1, 0): 0, (1, 2, 0): 1, (2, 3, 0): 2, (3, 4, 0): 3, (0, 4, 0): 4}
    nx.set_edge_attributes(g, seq, 'sequence')

    # initialize the edge order and get the labels
    eo = EdgeOrder(g)
    labels = eo.label_all_edges()

    # the labels that should have been produced
    expected = {(0, 1, 0): 1, (1, 2, 0): 2, (2, 3, 0): 3, (3, 4, 0): 4, (0, 4, 0): 5}

    assert path_works(g, labels)
    #assert labels == expected


def test_basic_path():
    """Test the labelling of edges in a path (non-closed cycle).

    This tests blocks where there is no connection from the start node to the end node.

    """

    # create a small graph and label the edges
    g = nx.path_graph(5, create_using=nx.MultiGraph)
    # edge order is going to look for a sequence field to determine the start node
    seq = {(0, 1, 0): 0, (1, 2, 0): 1, (2, 3, 0): 2, (3, 4, 0): 3}
    nx.set_edge_attributes(g, seq, 'sequence')

    # initialize the edge order and get the labels
    eo = EdgeOrder(g)
    labels = eo.label_all_edges()

    # the labels that should have been produced
    expected = {(0, 1, 0): 1, (1, 2, 0): 2, (2, 3, 0): 3, (3, 4, 0): 4}

    assert path_works(g, labels)
    #assert labels == expected


def test_interior_branch():
    """Test the labelling of edges in a cycle with an edge sticking off it.

    This tests that the order will traverse interior arcs when that arc does not create a closed loop.

    """

    # create a small graph and label the edges
    g = nx.cycle_graph(5, create_using=nx.MultiGraph)
    # add the branch edge
    g.add_edge(2,5)
    g.add_edge(5,2)
    # edge order is going to look for a sequence field to determine the start node
    seq = {(0, 1, 0): 0, (1, 2, 0): 1, (2, 5, 0): 2, (2, 5, 1): 3, (2, 3, 0): 4, (3, 4, 0): 5, (0, 4, 0): 6}
    nx.set_edge_attributes(g, seq, 'sequence')

    # initialize the edge order and get the labels
    eo = EdgeOrder(g)
    labels = eo.label_all_edges()

    # the labels that should have been produced
    expected = {(0, 1, 0): 1, (1, 2, 0): 2, (2, 5, 0): 3, (2, 5, 1): 4, (2, 3, 0): 5, (3, 4, 0): 6, (0, 4, 0): 7}

    assert path_works(g, labels)
    #assert labels == expected


def test_double_interior_edge():
    """Test the labelling of edges in a cycle with two edges sticking off it. One of the edges will also have a
    successor edge.

    This tests that the edge order follows along a path of interior arcs instead of expecting the person to
    zigzag across the road.

    """

    # create a small graph and label the edges
    g = nx.cycle_graph(5, create_using=nx.MultiGraph)

    # add the first branch edge
    g.add_edge(2,5)
    g.add_edge(5,2)

    # add the second branch edges
    g.add_edge(4, 6)
    g.add_edge(6, 4)
    g.add_edge(6, 7)
    g.add_edge(7, 6)

    # edge order is going to look for a sequence field to determine the start node
    seq = {(0, 1, 0): 0, (1, 2, 0): 1, (2, 5, 0): 2, (2, 5, 1): 3, (2, 3, 0): 4, (3, 4, 0): 5, (0, 4, 0): 6}
    nx.set_edge_attributes(g, seq, 'sequence')

    # initialize the edge order and get the labels
    eo = EdgeOrder(g)
    labels = eo.label_all_edges()

    # the labels that should have been produced
    expected = {(0, 1, 0): 1, (1, 2, 0): 2, (2, 5, 0): 3, (2, 5, 1): 4, (2, 3, 0): 5, (3, 4, 0): 6, (0, 4, 0): 11,
                (4, 6, 0): 7, (6, 7, 0): 8, (6, 7, 1): 9, (4, 6, 1): 10}

    assert path_works(g, labels)
    #assert labels == expected


def test_y_branch(caplog):
    """Test the labelling of edges in a cycle with an edge sticking off it that forms a Y with it's successors.

    This ensures that all branches of a set of interior arcs get processed before continuing along the border.

    """

    # capture debug logs on failures
    caplog.set_level(logging.DEBUG)

    # create a small graph and label the edges
    g = nx.cycle_graph(5, create_using=nx.MultiGraph)

    # add the first branch edge
    g.add_edge(2, 5)
    g.add_edge(5, 2)

    # add the second branch edges
    g.add_edge(4, 6)
    g.add_edge(6, 4)
    # first arm of the Y
    g.add_edge(6, 7)
    g.add_edge(7, 6)
    # second arm of the Y
    g.add_edge(6,8)
    g.add_edge(8,6)

    # edge order is going to look for a sequence field to determine the start node
    seq = {(0, 1, 0): 0, (1, 2, 0): 1, (2, 5, 0): 2, (2, 5, 1): 3, (2, 3, 0): 4, (3, 4, 0): 5, (0, 4, 0): 6}
    nx.set_edge_attributes(g, seq, 'sequence')

    # initialize the edge order and get the labels
    eo = EdgeOrder(g)
    labels = eo.label_all_edges()

    # the labels that should have been produced
    expected = {(0, 1, 0): 1, (1, 2, 0): 2, (2, 5, 0): 3, (2, 5, 1): 4,
                (2, 3, 0): 5, (3, 4, 0): 6, (4, 6, 0): 7, (6, 7, 0): 8, (6, 7, 1): 9,
                (6, 8, 0): 10, (6, 8, 1): 11, (4, 6, 1): 12,
                (0, 4, 0): 13}

    assert path_works(g, labels)
    #assert labels == expected


def test_interior_connecting_arc(caplog):
    """Test the labelling of edges where an interior arc connects two nodes but does not form a new block.

    In some instances an interior arc will connect nodes, resulting in a cyclic graph. This tests that the labelling
    traverses one side and comes back before traversing the rest of the block.

    """

    # capture debug logs on failure
    caplog.set_level(logging.DEBUG)

    g = nx.MultiGraph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 5)
    g.add_edge(5, 4)
    g.add_edge(4, 6)
    g.add_edge(6, 7)
    g.add_edge(7, 8)
    g.add_edge(8, 9)
    g.add_edge(9, 8)
    g.add_edge(9, 10)
    g.add_edge(10, 9)
    g.add_edge(9, 11)
    g.add_edge(11, 9)
    g.add_edge(8, 12)
    g.add_edge(12, 0)
    g.add_edge(12, 1)
    g.add_edge(1, 12)

    # edge order is going to look for a sequence field to determine the start node
    seq = {(0, 1, 0): 0, (1, 2, 0): 1, (2, 3, 0): 4, (3, 4, 0): 5, (0, 4, 0): 6}
    nx.set_edge_attributes(g, seq, 'sequence')

    # initialize the edge order and get the labels
    eo = EdgeOrder(g)
    labels = eo.label_all_edges()

    # the labels that should have been produced
    expected = {(0, 1, 0): 1, (1, 2, 0): 2, (2, 3, 0): 3, (3, 4, 0): 4,
                # small branch
                (4, 5, 0): 5, (4, 5, 1): 6,
                # back to border edges
                (4, 6, 0): 7, (6, 7, 0): 8, (7, 8, 0): 9,
                # Y branch
                (8, 9, 0): 10, (9, 10, 0): 11, (9, 10, 1): 12, (9, 11, 0): 13, (9, 11, 1): 14, (8, 9, 1): 15,
                # back on border again
                (8, 12, 0): 16, (1, 12, 0): 17, (1, 12, 1): 18, (0, 12, 0): 19}

    assert path_works(g, labels)
    #assert labels == expected


def test_nonzero_sequence_start():
    """Test the labelling of edges when the sequence number doesn't start with zero.

    It is possible that when the edges in a child geography are grouped that the starting edge will have a sequence
    value that is higher than zero.

    """

    # create a small graph and label the edges
    g = nx.cycle_graph(5, create_using=nx.MultiGraph)

    # add the first branch edge
    g.add_edge(2,5)
    g.add_edge(5,2)

    # add the second branch edges
    g.add_edge(4, 6)
    g.add_edge(6, 4)
    g.add_edge(6, 7)
    g.add_edge(7, 6)

    # edge order is going to look for a sequence field to determine the start node
    seq = {(0, 1, 0): 4, (1, 2, 0): 5, (2, 5, 0): 7, (2, 5, 1): 8, (2, 3, 0): 9, (3, 4, 0): 10, (0, 4, 0): 11}
    nx.set_edge_attributes(g, seq, 'sequence')

    # initialize the edge order and get the labels
    eo = EdgeOrder(g)
    labels = eo.label_all_edges()

    # the labels that should have been produced
    expected = {(0, 1, 0): 1, (1, 2, 0): 2, (2, 5, 0): 3, (2, 5, 1): 4, (2, 3, 0): 5, (3, 4, 0): 6, (0, 4, 0): 11,
                (4, 6, 0): 7, (6, 7, 0): 8, (6, 7, 1): 9, (4, 6, 1): 10}

    assert path_works(g, labels)
    #assert labels == expected


def test_disconnected_graph():
    """Test a basic graph that is disconnected.

    This is the same idea as an interior edge, but that edge is not connected to any of the border edges."""

    # create a small graph and label the edges
    g = nx.cycle_graph(5, create_using=nx.MultiGraph)
    # add the branch edge
    g.add_edge(6,5)
    g.add_edge(5,6)
    # edge order is going to look for a sequence field to determine the start node
    seq = {(0, 1, 0): 0, (1, 2, 0): 1, (6, 5, 0): 2, (6, 5, 1): 3, (2, 3, 0): 4, (3, 4, 0): 5, (0, 4, 0): 6}
    nx.set_edge_attributes(g, seq, 'sequence')

    # initialize the edge order and get the labels
    eo = EdgeOrder(g)
    labels = eo.label_all_edges()

    # the labels that should have been produced
    expected = {(0, 1, 0): 1, (1, 2, 0): 2, (6, 5, 0): 6, (6, 5, 1): 7, (2, 3, 0): 3, (3, 4, 0): 4, (0, 4, 0): 5}

    assert path_works(g, labels)
    #assert labels == expected


def test_interior_donut(caplog):
    """Test when a block is formed of a single edge that connects to itself, forming the interior circle of a donut."""

    # capture debug logs on failure
    caplog.set_level(logging.DEBUG)

    # create a single edge cycle graph
    g = nx.cycle_graph(1, create_using=nx.MultiGraph)
    # edge order is going to look for a sequence field to determine the start node
    seq = {(0, 0, 0): 0}
    nx.set_edge_attributes(g, seq, 'sequence')

    # initialize the edge order and get the labels
    eo = EdgeOrder(g)
    labels = eo.label_all_edges()

    # the expected labels
    expected = {(0,0,0): 1}

    assert path_works(g, labels)
    #assert labels == expected


def test_parallel_interior_edges(caplog):
    """Test a block that has a parallel road running alongside the boundary edge.

    Also includes a crossing interior arc, effectively creating two parallel edge sets.
    """

    # capture debug logs on failure
    caplog.set_level(logging.DEBUG)

    # design the graph
    g = nx.MultiGraph()
    # first set of edges
    g.add_edges_from([(0,1), (1,2), (2,3)])
    # second set of edges
    g.add_edges_from([(4,5), (5,6), (6,7)])
    g.add_edges_from([(5,4), (6,5), (7,6)])
    # third set
    g.add_edges_from([(8,9), (9,10)])
    # add interconnections
    g.add_edges_from([(0,4), (4,8)])
    g.add_edges_from([(1,5), (5,1)])
    g.add_edges_from([(3,7), (7,10)])
    g.add_edges_from([(2,6), (6,9), (6,2), (9,6)])
    # edge order is going to look for a sequence field to determine the start node
    seq = {(0, 1, 0): 0, (1, 2, 0): 1}
    nx.set_edge_attributes(g, seq, 'sequence')

    # initialize the edge order and get the labels
    eo = EdgeOrder(g)
    labels = eo.label_all_edges()

    # the expected labels
    expected = {(0, 1, 0): 1, (1,5,0): 2, (5,4,0): 3, (4,5,1): 4, (5,6,0): 5, (6,9,0): 6, (6,9,1): 7, (6,7,0): 8,
                (6,7,1): 9, (6,2,0): 10, (2,6,1): 11, (6,5,1): 12, (5,1,1): 13, (1,2,0): 14, (2,3,0): 15, (3,7,0): 16,
                (7,10,0): 17, (10,9,0): 18, (9,8,0): 19, (8,4,0): 20, (4,0,0): 21}

    assert path_works(g, labels)
    #assert labels == expected


def test_y_branch_start_at_base_of_tree(caplog):
    """Test the labelling of edges in a cycle with an edge sticking off it that forms a Y with it's successors.

    This ensures that all branches of a set of interior arcs get processed before continuing along the border.

    """

    # capture debug logs on failures
    caplog.set_level(logging.DEBUG)

    # create a small graph and label the edges
    g = nx.cycle_graph(5, create_using=nx.MultiGraph)

    # add the first branch edge
    g.add_edge(2, 5)
    g.add_edge(5, 2)

    # add the second branch edges
    g.add_edge(4, 6)
    g.add_edge(6, 4)
    # first arm of the Y
    g.add_edge(6, 7)
    g.add_edge(7, 6)
    # second arm of the Y
    g.add_edge(6,8)
    g.add_edge(8,6)

    # edge order is going to look for a sequence field to determine the start node
    seq = {(4, 6, 0): 0, (6, 7, 0): 1, (6, 7, 1): 2, (6, 8, 0): 3, (6, 8, 1): 4, (4, 6, 1): 5}
    nx.set_edge_attributes(g, seq, 'sequence')

    # initialize the edge order and get the labels
    eo = EdgeOrder(g)
    labels = eo.label_all_edges()

    # the labels that should have been produced
    expected = {(0, 1, 0): 8, (1, 2, 0): 9, (2, 5, 0): 10, (2, 5, 1): 11,
                (2, 3, 0): 12, (3, 4, 0): 13, (4, 6, 0): 1, (6, 7, 0): 2, (6, 7, 1): 3,
                (6, 8, 0): 4, (6, 8, 1): 5, (4, 6, 1): 6,
                (0, 4, 0): 7}

    assert path_works(g, labels)
    #assert labels == expected


def test_internal_lollipop(caplog):
    """Test a cyclic graph with a non-block forming internal lollipop arc arrangement."""

    # capture debug logs on failures
    caplog.set_level(logging.DEBUG)

    # create the test graph
    g = nx.cycle_graph(5, create_using=nx.MultiGraph)
    # add the lollipop
    g.add_edges_from([(3,5), (5,3), (5,5), (5,5)])

    # edge order is going to look for a sequence field to determine the start node
    seq = {(0, 1, 0): 0, (1, 2, 0): 1}
    nx.set_edge_attributes(g, seq, 'sequence')

    # initialize the edge order and get the labels
    eo = EdgeOrder(g)
    labels = eo.label_all_edges()

    # the labels that should have been produced
    expected = {(0, 1, 0): 9, (0, 4, 0): 1, (1, 2, 0): 8, (2, 3, 0): 7, (3, 4, 0): 2,
                (3, 5, 0): 6, (3, 5, 1): 3, (5, 5, 0): 5, (5, 5, 1): 4}

    assert labels == expected


def test_internal_connections_with_cutout(caplog):
    """Test that a block is successfully traversed when it has an interior cutout along one of its internal arcs."""

    # capture debug logs on failures
    caplog.set_level(logging.DEBUG)

    # create the graph
    g = nx.cycle_graph(16, create_using=nx.MultiGraph)
    # add the internal arcs
    g.add_edges_from([(1, 4), (4, 1)])
    g.add_edges_from([(6, 18), (18, 6), (9, 18), (18, 9), (18, 13), (13, 18)])
    g.add_edges_from([(5, 17), (17, 5), (15, 16), (16, 15), (16, 17)])

    # edge order is going to look for a sequence field to determine the start node
    seq = {(0, 1, 0): 0, (1, 2, 0): 1}
    nx.set_edge_attributes(g, seq, 'sequence')

    # initialize the edge order and get the labels
    eo = EdgeOrder(g)
    labels = eo.label_all_edges()

    pprint.pprint(dict(sorted(labels.items(), key=lambda t: t[1])))

    # labels that should have been produced
    expected = {(0, 15, 0): 1, (15, 16, 1): 2, (15, 16, 0): 3, (17, 16, 0): 4, (5, 17, 1): 5, (5, 6, 0): 6, 
                (6, 18, 1): 7, (13, 18, 1): 8, (13, 18, 0): 9, (9, 18, 1): 10, (9, 18, 0): 11, (6, 18, 0): 12,
                (6, 7, 0): 13, (7, 8, 0): 14, (8, 9, 0): 15, (9, 10, 0): 16, (10, 11, 0): 17, (11, 12, 0): 18,
                (12, 13, 0): 19, (13, 14, 0): 20, (14, 15, 0): 21, (0, 1, 0): 22, (1, 4, 1): 23, (3, 4, 0): 24,
                (2, 3, 0): 25, (1, 2, 0): 26, (1, 4, 0): 27, (4, 5, 0): 28, (5, 17, 0): 29}

    assert labels == expected


def test_complicated_internal_arcs(caplog):
    """Test an overly complicated internal arc structure within a block."""

    # capture debug logs on failures
    caplog.set_level(logging.DEBUG)

    g = nx.cycle_graph(14, create_using=nx.MultiGraph)
    # add internal arcs to form small squares in the block - these are all double sided
    g.add_edges_from([(13, 14), (13, 14), (14, 1), (14, 1)])
    g.add_edges_from([(15, 2), (15, 2), (4, 15), (4, 15)])
    # add a separate set of arcs that significantly subdivide the orignal block
    g.add_edges_from([(11, 16), (11, 16)])
    g.add_edges_from([(16, 17), (16, 17)])
    g.add_edges_from([(17, 5), (17, 5)])
    g.add_edges_from([(9, 18), (9, 18)])
    g.add_edges_from([(18, 16), (18, 16)])
    g.add_edges_from([(17, 19), (17, 19)])
    g.add_edges_from([(19, 8), (19, 8)])
    g.add_edges_from([(18, 19), (18, 19)])
    g.add_edges_from([(19, 6), (19, 6)])
    # break the cycle a little bit to make things more complicated
    g.remove_edges_from([(8, 9), (9, 10)])

    # edge order is going to look for a sequence field to determine the start node
    seq = {(0, 1, 0): 0, (1, 2, 0): 1}
    nx.set_edge_attributes(g, seq, 'sequence')

    # initialize the edge order and get the labels
    eo = EdgeOrder(g)
    labels = eo.label_all_edges()

    pprint.pprint(sorted(labels.items(), key=lambda t: t[1]))

    # labels that should have been produced
    expected = {(0, 1, 0): 27, (0, 13, 0): 26, (1, 2, 0): 28, (1, 14, 0): 24, (1, 14, 1): 23, (2, 3, 0): 33,
                (2, 15, 0): 32, (2, 15, 1): 29, (3, 4, 0): 34, (4, 5, 0): 35, (4, 15, 0): 31, (4, 15, 1): 30,
                (5, 6, 0): 36, (5, 17, 0): 17, (5, 17, 1): 16, (6, 7, 0): 37, (6, 19, 0): 10, (6, 19, 1): 9,
                (7, 8, 0): 38, (8, 19, 0): 8, (8, 19, 1): 7, (9, 18, 0): 13, (9, 18, 1): 12, (10, 11, 0): 1,
                (11, 12, 0): 20, (11, 16, 0): 19, (11, 16, 1): 2, (12, 13, 0): 21, (13, 14, 0): 25, (13, 14, 1): 22,
                (16, 17, 0): 18, (16, 17, 1): 15, (16, 18, 0): 14, (16, 18, 1): 3, (17, 19, 0): 6, (17, 19, 1): 5,
                (18, 19, 0): 11, (18, 19, 1): 4}


    assert len(labels) == len(g.edges)
    assert labels == expected