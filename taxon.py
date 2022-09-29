import numpy as np


class taxNode:

    # TODO make proper init later, this is just to brainstorm class variables
    def __init__(self):
        self.children = []     # list of children taxnodes
        self.prior = 1         # prior prob (np double?)
        self.prob = None       # probability of this outcome
        self.layer = 0         # depth of node in tree
        self.u_z = 1           # number of expected missing branches under this node

    def add_children(self, c):
        self.children.extend(c)
        for ch in c:
            ch.layer += 1
