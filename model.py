import numpy as np
import typing


class Protax:

    # TODO make proper constructor
    def __init__(self):
        self.params = None                   # betas
        self.q = 0                           # mislabeling rate
        self.predictor = seq_dist_predictor  # sequence dist based predictor

    def get_branch_prob(self, node, X):
        """
        return probability vector for each branch under provided node.
        Assume first row in X corresponds to unknown case.
        """

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
        if len(node.children) == 0:
            return

        X = self.predictor(node, query)
        branch_p = self.get_branch_prob(node, X)

        # TODO can this be vectorized? slow version for now
        for c in range(0, len(node.children)):
            node.children[c].prob = node.children[c].prior*self.q + node.prob*branch_p[c]*(1-self.q)
            self.classify(node.children[c], query)

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


def seq_dist_predictor(node, query):
    """
    sequence based predictor for generating X to feed into get_branch_prob
    """
    nz = len(node.children)
    res = np.zeros((nz, 4))

    # TODO format X following author's C code
    if nz == 1:
        return res

    mins = np.array([])
    for i, c in enumerate(node.children[1:], start=1):
        # no reference sequences
        if len(c.ref_seqs) == 0:
            res[i, 0] = 1

        # has reference sequences
        else:
            dists = np.array([seq_dist(r, query) for r in c.ref_seqs])
            res[i, 0:2] = 1
            if len(c.ref_seqs) == 1:
                res[i, 2] = dists[0]
                res[i, 3] = 1.0
            else:
                res[i, 2:4] = np.partition(mins, dists)[:2]

    return res
