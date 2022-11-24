import jax
import jax.numpy as jnp
import time
from functools import partial
from tree import TaxTree, HashableArrayWrapper


def classify(query, tree, beta, scalings):
    """
    Compute probability of all outcomes for this taxonomic tree rooted
    at the node in nid
    """
    R = tree.refs.shape[0]             # num references
    N = tree.children.shape[0]         # num nodes
    dists = seq_dist(query, tree.refs, tree.ref_lens)

    while tree.visited < N:
        curr = tree.visit_q.at[tree.q_end-1].get()
        
        # getting relevant children and reference indices
        curr_childs = jnp.unpackbits(tree.children.take(curr, axis=0))
        n_c = tree.num_ch.at[curr].get()
        child_idx = jnp.argwhere(curr_childs, size=n_c).flatten()
        child_unk = jnp.take(tree.unk, child_idx)
        c_refs = jnp.unpackbits(tree.node_refs.take(child_idx, axis=0), axis=1)[:, :R]

        # get 2 min distances
        has_refs = jnp.sum(c_refs, axis=1) > 0
        X = min_two_batch(c_refs, dists)
        X = jnp.concatenate((jnp.ones((X.shape[0], 1)), X), axis=1)

        # keep only distances of nodes with >1 refs and known
        X = jnp.multiply(X.T, has_refs)
        X = jnp.concatenate((jnp.ones((1, X.shape[1])), X), axis=0)
        X = jnp.multiply(X, jnp.logical_not(child_unk)).T
        
        # compute probability of children
        compute_probs(X, beta, tree, curr, scalings, child_idx)

        # add children to queue
        tree.visit_q = tree.visit_q.at[tree.q_end:tree.q_end+n_c].set(child_idx)
        tree.q_end = tree.q_end + n_c
        print(tree.visited)
        tree.visited += 1



def get_child_idx(child_adj, n_c):
    child_idx = jnp.argwhere(child_adj, size=n_c).flatten()
    return child_idx

jitted = jax.jit(get_child_idx, static_argnums=(1))

# jax.jit
def min_two(refs, dists):
    """
    minimum 2 seq distance predictor, given a node's references and
    all sequence distances.

    if no references return [1, 1]
    in only 1 reference return [min1, min1]
    Otherwise return [min1, min2]
    """
    valid_dists = dists.take(jnp.argwhere(refs, size=refs.shape[0]))
    valid_dists = jnp.append(valid_dists, 1)
    valid_dists_reversed = jnp.multiply(valid_dists, -1)

    # take 2 minimum indices
    _, min_inds = jax.lax.top_k(valid_dists_reversed, 1 + valid_dists.shape[0] >=2)
    min_inds = jnp.append(min_inds, min_inds.at[0].get())

    # return corresponding distances
    return valid_dists.take(min_inds).at[:2].get()

min_two_batch = jax.vmap(min_two, (0, None), 0)


# @jax.jit
def compute_probs(X, beta, tree, nid, scalings, c_idx):
    # get info about node
    lvl = tree.layer.at[nid].get()
    beta = beta.at[lvl].get()

    # apply bayes rule
    b_probs = branch_prob(X, beta.T)
    b_probs = jnp.multiply(b_probs, tree.prob.at[nid].get())
    b_probs = jnp.multiply(b_probs, tree.prior.take(c_idx))
    b_probs = jnp.divide(b_probs, jnp.sum(b_probs))

    # set probability of children
    tree.prob = jnp.insert(tree.prob, c_idx, b_probs)


@jax.jit
def get_predictors(q: jnp.ndarray, nid, tree, params, row, ref_i, X):
    """
    Get the predictors of a single taxon node.
    Assume this node has >0 reference sequences
    """

    # getting reference sequences of this node
    refs = tree["refs"].take(ref_i, axis=0)
    ref_lens = tree["ref_lens"].take(ref_i)

    # get 2 min dists
    dists = seq_dist(q, refs, ref_lens)
    d_vals, d_inds = jax.lax.top_k(jnp.multiply(dists, -1), int(dists.size > 0) + int(dists.size >= 2))
    d1 = d_inds.at[0].get()
    d_inds = jnp.append(d_inds, d1)
    d2 = d_inds.at[1].get()

    # return row
    X = X.at[row, 0:2].set(1)
    X = X.at[row, 2].set(dists.at[d1].get())
    X = X.at[row, 3].set(dists.at[d2].get())
    return X


def branch_prob(X, beta, u=1):
    """
    Computes the probability vector of each branch contained in X
    given model parameters and the expected number of unknown taxa
    in X.

    X: predictors for each branch under a node
    beta: regression parameters
    u: expected number of unknown species under the node
    """

    # applying weights
    z = jnp.exp(jnp.dot(X, beta))
    z = z.at[0].set(z.at[0].get()*u)
    norm_factor = jnp.sum(z)
    return jnp.divide(z, norm_factor)


def seq_dist(q, seqs, sizes):
    """
    Computes sequence distance between the query and an array of reference sequences
    """
    matches = jnp.bitwise_and(q, seqs)
    match_tots = jnp.sum(jnp.unpackbits(matches, axis=1), axis=1)
    return (sizes - match_tots) / sizes


if __name__ == '__main__':
    print(jax.devices())

