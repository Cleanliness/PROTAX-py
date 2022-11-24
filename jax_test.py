import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, treedef_children, treedef_tuple, tree_map
import chex


@chex.dataclass
class NodeTest():
    refs: chex.ArrayDevice
    n_refs: list
    children: list


@jax.jit
def test_jit(nid, b):
    h = tree_map(lambda x: b.refs.take(x), b.n_refs[nid])
    return h

bbb = NodeTest(
    refs=jnp.array([i*10 for i in range(4)]),
    n_refs=[jnp.array([1,2,3]), jnp.array([1]), jnp.array([1]), jnp.array([1])],
    children=[jnp.array([1, 3]), jnp.array([2]), jnp.array([]), jnp.array([])]
)

sd = test_jit(0, bbb)
sd = test_jit(0, bbb)
bb = 3