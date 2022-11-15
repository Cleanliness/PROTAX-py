import jax
import jax.numpy as jnp
import numpy as np
from taxon import *
import time
from collections import deque


def branch_prob(X, beta, u=1):
    """
    Computes the probability vector of each branch contained in X
    given model parameters and the expected number of unknown taxa
    in X.

    X: predictors for each branch under a node
    beta: regression parameters
    u: expected number of unknown species under the node
    """

    # applying weight
    z = jnp.exp(jnp.dot(X, beta))
    z = z.at[0].set(z.at[0].get()*u)
    norm_factor = jnp.sum(z)
    return jnp.divide(z, norm_factor)


def classify(query, nid, params, tree):
    """
    Compute probability of all outcomes for this taxonomic tree rooted
    at the node in nid
    """
    remaining_nodes = deque()
    remaining_nodes.append(nid)

    while len(remaining_nodes) > 0:
        curr = remaining_nodes.popleft()
        # getting relevant children and reference indices
        start_time = time.time()
        children = tree["children"][curr]
        print("Indexing took " + str(time.time() - start_time))

        # computing probability of all children nodes
        if len(children) > 0:
            X = compute_predictors(query, params, tree, curr, children)
            compute_probs(X, params, tree, nid, children)

        # add children to queue
        remaining_nodes.extend(children)


def compute_predictors(query, params, tree, nid, children):
    # get predictors for children under this node
    start_time = time.time()
    X = jnp.ones((len(children), 4))
    print("X creation took " + str(time.time() - start_time))

    # get probabilities of each child
    start_time = time.time()
    for i, c in enumerate(children):
        refs = tree["node_refs"][c]
        if tree["unk"].at[nid].get():
            X = X.at[i, :].set(0)
        elif refs.size == 0:
            X = X.at[i, 0:].set(0)
        else:
            X = get_predictors(query, c, tree, params, i, refs, X)
    print("Predictor computation took " + str(time.time() - start_time))
    return X

@jax.jit
def compute_probs(X, params, tree, nid, children):
    # apply bayes rule
    start_time = time.time()
    b_probs = branch_prob(X, params["beta"][tree["layer"][nid]].T)
    b_probs = jnp.multiply(b_probs, tree["prob"].at[nid].get())
    b_probs = jnp.multiply(b_probs, tree["prior"].take(children))
    b_probs = jnp.divide(b_probs, jnp.sum(b_probs))
    print("probability computation took " + str(time.time() - start_time))

    # set probability of children
    start_time = time.time()
    tree["prob"] = jnp.insert(tree["prob"], children, b_probs)
    print("probability assignment took " + str(time.time() - start_time))


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


def seq_dist(q, seqs, sizes):
    """
    Computes sequence distance between the query and an array of reference sequences
    """
    matches = jnp.bitwise_and(q, seqs)
    match_tots = jnp.sum(jnp.unpackbits(matches, axis=1), axis=1)
    return (sizes - match_tots) / sizes


if __name__ == '__main__':
    pass



