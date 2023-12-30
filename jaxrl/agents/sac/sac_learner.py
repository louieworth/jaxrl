"""Implementations of algorithms for continuous control."""
import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import jaxpruner

from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.actor import update as update_actor
from jaxrl.agents.sac.critic import target_update
from jaxrl.agents.sac.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.utils import calculate
from jaxrl.networks.common import InfoDict, Model, PRNGKey


# @functools.partial(jax.jit,
#                    static_argnames=('backup_entropy', 'update_target'))
def _update_jit(
    rng: PRNGKey, actor: Model, actor_pruner, critic: Model, critic_pruner: Model, 
    target_critic: Model, temp: Model, batch: Batch, discount: float, tau: float,
    target_entropy: float, backup_entropy: bool, update_target: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:
    
    # TODO: 他是直接apply mask还是通过直接改变weight得方法dynamic weight rescaling (layer normalization)
    rng, key = jax.random.split(rng)
    is_ste = isinstance(actor_pruner, (jaxpruner.SteMagnitudePruning,
                                         jaxpruner.SteRandomPruning))
    if is_ste:
        critic_pre_op = jax.jit(actor_pruner.pre_forward_update)   
        critic_new_params = critic_pre_op(actor.params, actor.opt_state)
        critic = critic.replace(params=critic_new_params)
        
        actor_pre_op = jax.jit(actor_pruner.pre_forward_update)   
        actor_new_params = actor_pre_op(actor.params, actor.opt_state)
        actor = actor.replace(params=actor_new_params)
        
    new_critic, critic_info, critic_grad_fn = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            backup_entropy=backup_entropy)
    # TODO apply mask一般是post_gradient_update 还是 pre_forward_update？有什么样的区别？什么时候是apply mask，什么时候是apply gradient？
    # Apply masks to parameters.
    post_critic_params = critic_pruner.post_gradient_update(
        new_critic.params, new_critic.opt_state)
    new_critic = new_critic.replace(params=post_critic_params)

    rng, key = jax.random.split(rng)
    new_actor, actor_info, actor_grad_fn = update_actor(key, actor, new_critic, temp, batch)
    post_actor_params = actor_pruner.post_gradient_update(
    new_actor.params, new_actor.opt_state)
    new_actor = new_actor.replace(params=post_actor_params)
    
    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy)
    
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic
    
    sparsity_info = {}
    if is_ste:
        actor_sparsity = jaxpruner.summarize_sparsity(actor_new_params)
        critic_sparsity = jaxpruner.summarize_sparsity(critic_new_params)
    else:
        actor_sparsity = jaxpruner.summarize_sparsity(new_actor.params)
        critic_sparsity = jaxpruner.summarize_sparsity(new_critic.params)
    sparsity_info['actor_sparsity'] = actor_sparsity['_total_sparsity']
    sparsity_info['critic_sparsity'] = critic_sparsity['_total_sparsity']
    return rng, new_actor, new_critic, new_target_critic, new_temp, critic_grad_fn, actor_grad_fn,{
        **critic_info,
        **actor_info,
        **alpha_info,
        **sparsity_info
    }


class SACLearner(object):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 backup_entropy: bool = True,
                 init_temperature: float = 1.0,
                 init_mean: Optional[np.ndarray] = None,
                 policy_final_fc_init_scale: float = 1.0,
                 layer_normalization: bool = False,
                 actor_pruner: jaxpruner.algorithms = None,
                 critic_pruner: jaxpruner.algorithms = None):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        actor_def = policies.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            init_mean=init_mean,
            layer_normalization=layer_normalization,
            final_fc_init_scale=policy_final_fc_init_scale)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=actor_pruner.wrap_optax(optax.adam(learning_rate=actor_lr)))

        critic_def = critic_net.DoubleCritic(hidden_dims, layer_normalization=layer_normalization)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=critic_pruner.wrap_optax(optax.adam(learning_rate=critic_lr)))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        copy_actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        copy_critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=actor_lr))
        
        last_actor = Model.create(actor_def,
                             inputs=[actor_key, observations])

        last_critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions])
        copy_target_critic = Model.create(critic_def, 
                                     inputs=[critic_key, observations, actions])


        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=temp_lr))

        self.actor_pruner = actor_pruner
        self.critic_pruner = critic_pruner
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng
        
        self.copy_actor = copy_actor
        self.copy_critic = copy_critic
        self.copy_target_critic = copy_target_critic
        
        self.last_actor = last_actor
        self.last_critic = last_critic
        self.critic_grad = None
        self.actor_grad = None

        self.step = 1

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
        self.step += 1

        new_rng, new_actor, new_critic, new_target_critic, new_temp, new_critic_grad, new_actor_grad, info = _update_jit(
            self.rng, self.actor, self.actor_pruner, self.critic,  self.critic_pruner, 
            self.target_critic, self.temp, batch, self.discount, self.tau, self.target_entropy,
            self.backup_entropy, self.step % self.target_update_period == 0)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp
        
        if self.critic_grad is None:
            self.critic_grad = new_critic_grad
            self.actor_gead = new_actor_grad

        return info, new_critic_grad, new_actor_grad

    def update_grad_info(self, new_grad, is_actor) -> InfoDict:

        if is_actor:
            actor_grad_info = calculate(new_grad, self.actor_grad, is_actor=True)
        else:
            critic_grad_info= calculate(new_grad, self.critic_grad, is_actor=False)
        
        self.critic_grad = new_grad
        self.actor_grad = new_grad
         
        return {
            **actor_grad_info,
            **critic_grad_info
        }