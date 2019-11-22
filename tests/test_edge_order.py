import networkx as nx

from blocksequence.blocksequence import EdgeOrder

def test_basic_cycle():
    """Test the labelling of edges in a small cycle."""

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
    """Test the labelling of edges in a path (non-closed cycle)."""

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
    """Test the labelling of edges in a cycle with an edge sticking off it."""

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