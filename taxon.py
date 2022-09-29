import numpy as np

class taxNode:

    # TODO make proper init later, this is just to brainstorm class variables
    def __init__(self):
        self.children = []     # list of children taxnodes
        self.X = np.array([])  # predictors for each child
        self.prior = 1         # prior prob (np double?)
        self.prob = 1          # probability of this outcome