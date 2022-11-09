from taxon import taxNode
from model import *
import numpy as np
import protax_utils
import pytest

"""
Some basic sanity checks on model classification, and 
distance computation
"""


def test_seq_dist_vectorized():
    seqs = ["TTTTTT", "GGGGGG", "CCCCCC", "AAAGGG"]
    seq_bit = []
    for s in seqs:
        seq_bit.append(protax_utils.get_seq_bits(s))

    lens = np.array([6, 6, 6, 6])
    q = protax_utils.get_seq_bits("ATGGCG")
    res = seq_dist_vectorized(q, np.array(seq_bit), lens)
    assert np.all(res == np.array([5/6, 0.5, 5/6, 0.5]))


def test_bseq_dist_match():
    """
    sequence distance for completely matching sequences
    """
    a = protax_utils.get_seq_bits("AAAAAAAAAAA")
    b = protax_utils.get_seq_bits("AAAAAAAAAAA")

    d = seq_dist_bitw(a, b, 11)
    assert d == 0


def test_bseq_dist_no_match():
    """
    sequence distance for completely matching sequences
    """
    a = protax_utils.get_seq_bits("AAAAAAAAAAA")
    b = protax_utils.get_seq_bits("TTTTTTTTTTT")

    d = seq_dist_bitw(a, b, 11)
    assert d == 1.0


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
    ref_seqs = ["TTTTTT", "GGGGGG", "CCCCCC", "AAAGGG"]
    m.set_reference_sequences([protax_utils.get_seq_bits(s) for s in ref_seqs],
                              [6 for i in range(7)])
    sc = np.ones((2, 4))
    sc[:, [0, 2]] = 0
    m.set_scaling(sc)
    root = make_small_tree()

    m.classify(root, protax_utils.get_seq_bits("ATAGCG"))
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

    ref_seqs = ["TTTTTT", "GGGGGG", "CCCCCC", "AAAGGG", "ATGCGA", "ATGCCC", "ATATTA"]
    m.set_reference_sequences([protax_utils.get_seq_bits(s) for s in ref_seqs],
                              [6 for i in range(7)])

    sc = np.ones((2, 4))
    sc[:, [0, 2]] = 0
    m.set_scaling(sc)

    root = make_med_tree()
    m.classify(root, protax_utils.get_seq_bits("ATAGCG"))
    res = root.get_probs()

    assert sum(res) == 3


# ========= Helpers ===========
def make_small_tree():

    root = taxNode()
    root.ref_indices.append(3)
    root.ref_indices = np.array(root.ref_indices)
    root.prior = 1
    root.prob = 1

    for i in range(4):
        curr = taxNode()
        curr.prior = 0.25
        curr.ref_indices.append(i)
        curr.ref_indices = np.array(curr.ref_indices)
        root.add_children([curr])

    return root


def make_med_tree():
    root = taxNode()
    root.ref_indices.append(3)
    root.ref_indices = np.array(root.ref_indices)
    root.prior = 1
    root.prob = 1

    children = []

    for i in range(4):
        curr = taxNode()
        curr.prior = 0.25
        children.append(curr)
        curr.ref_indices.append(i)
        curr.ref_indices = np.array(curr.ref_indices)
        root.add_children([curr])

    for c in children:
        for cc in range(4, 7):
            curr = taxNode()
            curr.prior = 0.0625
            curr.ref_indices.append(cc)
            curr.ref_indices = np.array(curr.ref_indices)
            c.add_children([curr])

    return root


if __name__ == '__main__':
    pytest.main(['test.py'])