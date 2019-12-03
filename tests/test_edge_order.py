import logging
import networkx as nx

from blocksequence.blocksequence import EdgeOrder

def test_basic_cycle():
    """Test the labelling of edges in a small cycle.

    This tests that the order will follow a basic cycle when the block arcs all connect in a simple circle.

    """

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

    assert labels == expected


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

    assert labels == expected


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

    assert labels == expected


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

    assert labels == expected


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

    assert labels == expected


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

    assert labels == expected


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

    assert labels == expected


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

    assert labels == expected