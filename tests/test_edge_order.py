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
    labels = eo.label_edges()

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
    labels = eo.label_edges()

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
    labels = eo.label_edges()

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
    labels = eo.label_edges()

    # the labels that should have been produced
    expected = {(0, 1, 0): 1, (1, 2, 0): 2, (2, 5, 0): 3, (2, 5, 1): 4, (2, 3, 0): 5, (3, 4, 0): 6, (0, 4, 0): 11,
                (4, 6, 0): 7, (6, 7, 0): 8, (6, 7, 1): 9, (4, 6, 1): 10}

    assert labels == expected


def test_y_branch():
    """Test the labelling of edges in a cycle with an edge sticking off it that forms a Y with it's successors.

    This ensures that all branches of a set of interior arcs get processed before continuing along the border.

    """

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
    labels = eo.label_edges()

    # the labels that should have been produced
    expected = {(0, 1, 0): 1, (1, 2, 0): 2, (2, 5, 0): 3, (2, 5, 1): 4,
                (2, 3, 0): 5, (3, 4, 0): 6, (4, 6, 0): 7, (6, 7, 0): 8, (6, 7, 1): 9,
                (6, 8, 0): 10, (6, 8, 1): 11, (4, 6, 1): 12,
                (0, 4, 0): 13}

    assert labels == expected