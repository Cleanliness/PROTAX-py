from taxon import taxNode
from model import Protax, seq_dist
import numpy as np
import pytest


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


if __name__ == '__main__':
    pytest.main(['test.py'])
