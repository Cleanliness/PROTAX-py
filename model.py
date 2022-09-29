import numpy as np
import typing


class Protax:

    # TODO make proper constructor
    def __init__(self):
        self.params = None          # betas
        self.q = -1                 # mislabeling rate

    def get_branch_prob(self, node, X):
        """
        return probability vector for each branch under a node.
        """

        # TODO check for correctness vs C protax
        # get weighted sums
        beta = self.params[node.layer]
        n_z, m = X.shape
        exp_z = np.exp(X @ beta)

        # apply weights in softmax computation
        weight_mask = np.ones(n_z)
        weight_mask[0] = node.u_z
        norm_factor = weight_mask @ exp_z  # TODO prevent div by 0
        w_exp_z = exp_z * weight_mask

        return w_exp_z / norm_factor

    def classify(self, node, query):
        """
        return probability vector of each outcome.
        """
        
        # TODO compute X
        branch_p = self.get_branch_prob(node, query)

        # TODO can this be vectorized? slow version for now
        for c in range(0, len(node.children)):
            node.children[c].prob = node.prob*branch_p[c]
            self.classify(node.children[c], query)
        
        if len(node.children) == 0:
            return node.prior*self.q + (1-self.q)*node.prob

    def set_params(self, b, q):
        self.params = b
        self.q = q


# ============= MISC functions ===============
def seq_dist(a, b):
    """
    sequence distance between a and b. Lengths must match
    """
    ok_positions = 0
    mismatches = 0

    for i in range(len(a)):
        if a[i] in "ATGC" and b[i] in "ATGC":
            ok_positions += 1
            mismatches += int(a[i] != b[i])
    
    if ok_positions == 0:
        return 1.0
    return mismatches/ok_positions


def get_predictors(self, node, query):
    """
    Compute X to feed into get_branch_prob
    """
    nz = len(node.children)
    m = 3
    res = np.zeros((nz, m))

    return res
