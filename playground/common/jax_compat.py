"""Shims for jax APIs removed in jax 0.10 that brax still calls."""
import jax
import jax.numpy as jnp
import numpy as np


def _sharding(devices, name):
    mesh = jax.sharding.Mesh(np.array(devices), (name,))
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(name))


def _device_put_replicated(tree, devices):
    sharding = _sharding(devices, "_dpr")
    n = len(devices)
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.stack([jnp.asarray(x)] * n), sharding), tree
    )


def _device_put_sharded(shards, devices):
    sharding = _sharding(devices, "_dps")
    return jax.tree_util.tree_map(
        lambda *xs: jax.device_put(jnp.stack([jnp.asarray(x) for x in xs]), sharding),
        *shards,
    )


if not hasattr(jax, "device_put_replicated"):
    jax.device_put_replicated = _device_put_replicated
if not hasattr(jax, "device_put_sharded"):
    jax.device_put_sharded = _device_put_sharded
