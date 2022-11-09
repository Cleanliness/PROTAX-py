import numpy as np
import typing
import time


class Protax:

    # TODO make proper constructor
    def __init__(self):
        self.params = None                        # betas
        self.scaling = None                       # scaling factors
        self.q = 0                                # mislabeling rate
        self.predictor = self.seq_dist_predictor  # sequence dist based predictor
        self.refs = []                            # reference sequence bit vectors
        self.ref_lengths = []                     # lengths of all reference sequences

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
        computes probability of each outcome, stored in node.
        """
        if len(node.children) == 0:
            return

        X = self.predictor(node, query)  # slowest step
        branch_p = self.get_branch_prob(node, X)
        csum = 0.0

        # get joint probability of all children
        for c in range(0, len(node.children)):
            node.children[c].prob = node.children[c].prior*branch_p[c]
            csum += node.children[c].prob

        # normalize, i.e apply marginalized joint prob over outcomes
        for c in range(0, len(node.children)):
            node.children[c].prob = node.children[c].prob*node.prob/csum
            self.classify(node.children[c], query)

    def set_params(self, b, q):
        self.params = b
        self.q = q

    def set_scaling(self, s):
        self.scaling = s

    def set_reference_sequences(self, refs, lens):
        self.refs = refs
        self.ref_lengths = lens

    def seq_dist_predictor(self, node, query):
        """
        sequence based predictor for generating X to feed into get_branch_prob
        """
        nz = len(node.children)
        res = np.zeros((nz, 4))
        sc = self.scaling[node.layer]

        if nz == 1:
            return res

        for i, c in enumerate(node.children[1:], start=1):
            # no reference sequences
            if len(c.ref_indices) == 0:
                res[i, 0] = 1

            # has reference sequences
            else:
                # start_time = time.time()
                c_refs = np.take(self.refs, c.ref_indices, axis=0)
                # print("--- %s seconds to unpack ---" % (time.time() - start_time), end='\n')

                # start_time = time.time()
                lens = np.take(self.ref_lengths, c.ref_indices)
                # print("--- %s seconds to take lengths ---" % (time.time() - start_time), end='\n')

                start_time = time.time()
                dists = seq_dist_vectorized(query, c_refs, lens)
                print("computed " + str(dists.size) + " distances in " + str(time.time() - start_time))

                res[i, 0:2] = 1
                if c.ref_indices.size == 1:
                    res[i, 2] = (dists[0] - sc[0])/sc[1]
                    res[i, 3] = 1.0
                else:
                    res[i, 2:4] = np.partition(dists, 2)[:2]
                    res[i, 2] = (res[i, 2]-sc[2])/sc[3]
                    res[i, 3] = 1.0

        return res


# ============= MISC functions ===============
def seq_dist(a, b):
    """
    sequence distance between sequences a and b. Lengths must match.
    a,b are bit vectors
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


def seq_dist_bitw(a, b, size):
    match = np.bitwise_and(a, b)
    match_tot = sum(np.unpackbits(match))
    return (size - match_tot) / size


def seq_dist_vectorized(q, seqs, sizes):
    matches = np.bitwise_and(q, seqs)
    match_tots = np.sum(np.unpackbits(matches, axis=1), axis=1)
    return (sizes - match_tots) / sizes
