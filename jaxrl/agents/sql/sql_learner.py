"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

# from jaxrl.agents.sql import temperature
from jaxrl.agents.sql.actor import update as update_actor
from jaxrl.agents.sql.critic import target_update
from jaxrl.agents.sql.critic import update_q, update_v, target_update
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey


@jax.jit
def _update_jit_sql(
    rng: PRNGKey, actor: Model, critic: Model,
    value: Model, target_critic: Model, batch: Batch, discount: float, tau: float,
    alpha: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    #SQL
    new_value, value_info = update_v(target_critic, value, batch, alpha, alg='SQL')
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, target_critic,
                                         new_value, batch, alpha, alg='SQL')
    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }

@jax.jit
def _update_jit_eql(
    rng: PRNGKey, actor: Model, critic: Model,
    value: Model, target_critic: Model, batch: Batch, discount: float, tau: float,
    alpha: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    #EQL
    new_value, value_info = update_v(target_critic, value, batch, alpha, alg='EQL')
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, target_critic,
                                         new_value, batch, alpha, alg='EQL')
    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }

class SQLLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                #  expectile: float = 0.8,
                 alpha: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 value_dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 max_clip: Optional[int] = None,
                 alg: Optional[str] = None,
                 opt_decay_schedule: str = "cosine"):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        # self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.alpha = alpha
        self.max_clip = max_clip
        self.alg = alg

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = policies.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optimiser)

        critic_def = critic_net.DoubleCritic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))

        value_def = critic_net.ValueCritic(hidden_dims, dropout_rate=value_dropout_rate)
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=optax.adam(learning_rate=value_lr))

        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                             self.actor.params, observations,
                                             temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        if self.alg == 'SQL':
            new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit_sql(
                self.rng, self.actor, self.critic, self.value, self.target_critic,
                batch, self.discount, self.tau, self.alpha)
        elif self.alg == 'EQL':
            new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit_eql(
                self.rng, self.actor, self.critic, self.value, self.target_critic,
                batch, self.discount, self.tau, self.alpha)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        return info
