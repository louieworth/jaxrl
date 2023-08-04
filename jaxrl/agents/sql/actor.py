from typing import Tuple

import jax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey

def update(key: PRNGKey, actor: Model, critic: Model, value: Model,
                 batch: Batch, alpha: float, alg: str) -> Tuple[Model, InfoDict]:
    v = value(batch.observations)
    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)

    weight = q - v
    weight = jnp.maximum(weight, 0)
    weight = jnp.clip(weight, 0, 100.)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(weight * log_probs).mean()
        return actor_loss, {'actor_loss': actor_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info