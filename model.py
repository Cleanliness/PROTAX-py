import numpy as np
import typing

class Protax:

    # TODO make proper constructor
    def __init__(self):
        self.params = np.array([[]])    # betas
        self.q = 0                      # mislabeling rate

    def get_branch_prob(self, X, beta):
        """
        return probability vector for each branch under a node.
        """

        # TODO check for correctness
        # TODO do proper type annotations
        n_z, m = X.shape
        z = np.exp(X @ beta)
        weight_mask = np.ones((n_z, 1))
        weight_mask[0] = n_z  # TODO what's the expected number of missing branches?

        normalization_factor = weight_mask @ z  # TODO prevent div by 0
        return z / normalization_factor


    def classify(self, node, query):
        """
        return probability vector of each outcome.
        """
        
        # TODO compute X
        branch_p = get_branch_prob(node.X)

        # TODO can this be vectorized? slow version for now
        for c in range(0, len(node.children)):
            node.children[c].prob = node.prob*branch_p[c]
            classify(node.children[c], query)
        
        if len(node.children) == 0:
            return node.prior*q + (1-q)*node.prob


# ============= MISC functions ===============

def seq_dist(a, b, seq_len):

    ok_positions = 0
    mismatches = 0

    for i in range(seq_len):
        if a[i] in "ATGC" and b[i] in "ATGC":
            ok_positions += 1
            mismatches += int(a[i] != b[i])
    
    if ok_positions == 0:
        return 1.0
    return mismatches/ok_positions