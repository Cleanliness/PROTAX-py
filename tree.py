import chex
from typing import Generic, TypeVar
import jax.numpy as jnp

T = TypeVar('T')      # Declare type variable

@chex.dataclass
class TaxTree():
    """
    State of the taxonomic tree

    Let N be the number of nodes, R be the number of sequences

    refs: All reference sequences
    ref_lens: length of each reference sequence at the same index
    node_refs: reference sequences belong to node at the same index
    layer: layer in the taxonomic tree this node is in
    prior: prior probability of each node
    prob: predicted probability of each node
    children: adjacency matrix of each node
    num_ch: number of children
    unk: Whether the node at this index represents an unknown species or not
    visit_q: Nodes to be visited
    q_end: Number of nodes in the queue
    visited: Number of nodes already visited
    """
    refs: chex.ArrayDevice           # [R, 4]
    ref_lens: chex.ArrayDevice       # [R]
    node_refs: chex.ArrayDevice      # [N, R] NOTE: these dims are after unpacking along axis 1
    layer: chex.ArrayDevice          # [N]
    prior: chex.ArrayDevice          # [N]
    prob: chex.ArrayDevice           # [N]
    children: chex.ArrayDevice       # [N, N] NOTE: these dims are after unpacking along axis 1
    num_ch: chex.ArrayDevice         # [N]
    unk: chex.ArrayDevice            # [N]
    visit_q: chex.ArrayDevice        # [N]
    q_end: int
    visited: int


class HashableArrayWrapper(Generic[T]):
    def __init__(self, val: T):
        self.val = val

    def __getattribute__(self, prop):
        if prop == 'val' or prop == "__hash__" or prop == "__eq__":
            return super(HashableArrayWrapper, self).__getattribute__(prop)
        return getattr(self.val, prop)

    def __getitem__(self, key):
        return self.val.at[key].get()

    def __setitem__(self, key, val):
        self.val.at[key].set(val)

    def __hash__(self):
        return int(jnp.sum(self.val))

    def __eq__(self, other):
        if isinstance(other, HashableArrayWrapper):
            return self.__hash__() == other.__hash__()

        f = getattr(self.val, "__eq__")
        return f(self, other)