import numpy as np


class taxNode:

    # TODO make proper init later, this is just to brainstorm class variables
    def __init__(self, prior=0.0, unk=False, name=""):
        self.children = []     # list of children taxnodes
        self.prior = prior     # prior prob (np double?)
        self.prob = 0.0        # probability of this outcome
        self.layer = 0         # depth of node in tree
        self.u_z = 1           # number of expected missing branches under this node
        self.unk = unk         # true if representing unknown node
        self.name = name       # name of tree
        self.ref_indices = []     #

    def add_children(self, c):
        self.children.extend(c)
        for ch in c:
            ch.layer += 1

    def add_child(self, c):
        self.children.append(c)
        c.layer += 1

    def get_probs(self):
        """
        Get probability of each descendant node in preorder as a string
        """
        res = [self.prob]
        if len(self.children) != 0:
            for t in self.children:
                res.extend(t.get_probs())

        return res

    def set_ref_seqs(self, ref_seqs):
        self.ref_indices = ref_seqs

    def __str__(self):
        return str(self.prob)

    def __repr__(self):
        return str(self.prob)
