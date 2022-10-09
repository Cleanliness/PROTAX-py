from taxon import taxNode
from model import Protax, seq_dist
import numpy as np
import pytest


def test_taxon_init():
    """
    Test proper initialization
    """
    root = taxNode()
    root.add_children([taxNode() for i in range(3)])

    assert root.get_probs() == [0.0, 0.0, 0.0, 0.0]


def test_branch_prob_valid():
    """
    Test branch probabilities are valid, and dimensions match
    """
    b = np.zeros((3, 3))
    q = 0.5
    m = Protax()
    m.set_params(b, q)

    root = taxNode()
    root.add_children([taxNode() for i in range(3)])

    res = m.get_branch_prob(root, np.ones((3, 3)))

    assert res.shape == (3,)
    assert np.max(res) <= 1.0
    assert np.min(res) >= 0.0
    assert np.sum(res) == 1.0


def test_branch_prob_single_child():
    """
    Test branch probabilities of node with a single child is valid
    """

    b = np.zeros((1, 3))
    q = 0.5
    m = Protax()
    m.set_params(b, q)

    root = taxNode()
    root.add_children([taxNode()])

    res = m.get_branch_prob(root, np.ones((1, 3)))

    assert res[0] == 1.0


def test_outcome_small():
    """
    Test outcome probabilities on a small tree are valid
    """
    b = np.ones((1, 4))
    q = 0.0
    m = Protax()
    m.set_params(b, q)

    root = make_small_tree()

    m.classify(root, "ATAGCG")
    res = root.get_probs()

    # because there are 2 layers in the tree
    assert sum(res) == 2


def test_outcome_med():
    """
    Test outcome probabilities are valid on a medium tree with 3 layers
    """
    b = np.ones((2, 4))
    q = 0.0
    m = Protax()
    m.set_params(b, q)

    root = make_med_tree()

    m.classify(root, "ATAGCG")
    res = root.get_probs()

    assert sum(res) == 3


def test_seq_dist_match():
    """
    sequence distance for completely matching sequences
    """
    a = "AAAAAAAAAAA"
    b = "AAAAAAAAAAA"

    d = seq_dist(a, b)
    assert d == 0


def test_seq_dist_no_match():
    """
    sequence distance for completely matching sequences
    """
    a = "AAAAAAAAAAA"
    b = "TTTTTTTTTTT"

    d = seq_dist(a, b)
    assert d == 1.0


# ========= Helpers ===========
def make_small_tree():
    root = taxNode()
    root.ref_seqs.append("AAAAAA")
    root.prior = 1
    root.prob = 1

    for i in ["TTTTTT", "GGGGGG", "CCCCCC", "AAAGGG"]:
        curr = taxNode()
        curr.prior = 0.25
        curr.ref_seqs.append(i)
        root.add_children([curr])

    return root


def make_med_tree():
    root = taxNode()
    root.ref_seqs.append("AAAAAA")
    root.prior = 1
    root.prob = 1

    children = []

    for i in ["TTTTTT", "GGGGGG", "CCCCCC", "AAAGGG"]:
        curr = taxNode()
        curr.prior = 0.25
        children.append(curr)
        curr.ref_seqs.append(i)
        root.add_children([curr])

    for c in children:
        for cc in ["ATGCGA", "ATGCCC", "ATATTA"]:
            curr = taxNode()
            curr.prior = 0.0625
            curr.ref_seqs.append(cc)
            c.add_children([curr])

    return root


if __name__ == '__main__':
    pytest.main(['test.py'])
