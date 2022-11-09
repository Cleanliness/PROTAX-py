import numpy as np

from typing import *
import jax.numpy as jnp
import numpy as np

class TaxNode:
    """Represents a node in a taxonomic tree"""

    def __init__(self, prior: jnp.float64, unk: jnp.int8 = 0, name: str = ""):
        self.children = None  # list of children tax nodes
        self.ref_indices = None  # indices of reference sequences
        self.prior = prior  # prior probability of this node
        self.prob = None  # predicted probability of this outcome
        self.layer = 0  # depth of node in tree
        self.u_z = 1  # number of expected missing branches under this node
        self.unk = unk  # 1 if representing unknown node
        self.name = name  # name of node

    def set_children(self, c: jnp.DeviceArray):
        """Set the children indices of this node"""
        self.children = c

    def set_ref_seqs(self, ref_seqs: jnp.DeviceArray):
        """
        Set reference sequences of this node
        """
        self.ref_indices = ref_seqs

    def __str__(self):
        return str(self.prob)

    def __repr__(self):
        return str(self.prob)


class TaxColl:
    """Collection of taxa"""

    def __init__(self, nodes: jnp.ndarray, seqs: jnp.ndarray):
        """Initialize this taxonomy collection"""
        self.nodes = nodes
        self.seqs = seqs

    def get_root(self):
        """Return root node in this taxonomy collection"""
        return self.nodes.at[0].get()

    def get_seqs(self, node: TaxNode):
        return self.seqs.take(self.nodes[node].ref_indices)


