import jax
import jax.numpy as jnp
from collections import deque
from functools import partial
from tree import TaxTree, HashableArrayWrapper
import numpy as np
from functools import partial

@partial(jax.jit, static_argnums=(4, 5))
def classify(query, tree, beta, scalings, N, R):
    """
    Compute probability of all outcomes for this taxonomic tree rooted
    at the node in nid
    """
    dists = seq_dist(query, tree.refs, tree.ref_lens)

    # getting design matrices for all nodes
    X = jnp.zeros((N, 2))
    def body_fun(i, val):
        res = get_min2(tree.node_refs.at[i].get(), dists, R)
        return val.at[i].set(res)

    X = jax.lax.fori_loop(0, N, body_fun, X)
    X = jnp.concatenate((jnp.ones((N, 1)), X), axis=1)
    X = jnp.multiply(X.T, tree.has_refs)
    X = jnp.concatenate((jnp.ones((1, N)), X), axis=0)
    X = jnp.multiply(X, jnp.logical_not(tree.unk)).T

    probs = tree.prob.at[:].get()

    # calculating probabilities of each node
    def body_fun2(i, probs):
        curr_childs = jnp.unpackbits(tree.children.at[i].get()).at[:N].get()
        lvl = tree.layer.at[i].get()
        curr_beta = beta.at[lvl].get()
        valid_x = jnp.multiply(X.T, curr_childs).T

        branch_probs = branch_prob(valid_x, curr_beta, curr_childs)
        c_probs = jnp.multiply(branch_probs, probs.at[i].get())
        probs = jnp.add(probs, c_probs)
        
        return probs
    probs = jax.lax.fori_loop(0, N, body_fun2, probs)
    return probs


def get_min2(refs, dists, R):
    """
    Compute the minimum two sequence distances given a boolean array
    mask indicating the references for some node
    """
    refs = jnp.unpackbits(refs).at[:R].get()
    refs_neg = jnp.multiply(jnp.logical_not(refs), 2)
    dists = jnp.multiply(refs, dists)
    dists_rev = jnp.multiply(jnp.add(dists, refs_neg), -1)
    _, min_inds = jax.lax.top_k(dists_rev, 2)

    res = dists.take(min_inds)
    return res

# this takes up too much memory don't use
min2_batch = jax.vmap(get_min2, (0, None, None), 0)


def branch_prob(X, beta, ch):
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
    z = jnp.multiply(z.T, ch).T
    norm_factor = jnp.sum(z)
    return jnp.nan_to_num(jnp.divide(z, norm_factor))


def seq_dist(q, seqs, sizes):
    """
    Computes sequence distance between the query and an array of reference sequences
    """
    matches = jnp.bitwise_and(q, seqs)
    match_tots = jnp.sum(jnp.unpackbits(matches, axis=1), axis=1)
    return (sizes - match_tots) / sizes


if __name__ == '__main__':
    print(jax.devices())

