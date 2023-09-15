"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple
import os
import logging

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
import numpy as np
import optax
from jaxrl.utils import make_env, Log, calculate

from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.actor import update as update_actor
from jaxrl.agents.sac.critic import target_update
from jaxrl.agents.sac.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey


@functools.partial(jax.jit,
                   static_argnames=('backup_entropy', 'update_target'))
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
    temp: Model, batch: Batch, discount: float, tau: float,
    target_entropy: float, backup_entropy: bool, update_target: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info, critic_grad_fn = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            backup_entropy=backup_entropy)
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    rng, key = jax.random.split(rng)
    new_actor, actor_info, actor_grad_fn = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_target_critic, new_temp, critic_grad_fn, actor_grad_fn, {
        **critic_info,
        **actor_info,
        **alpha_info
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
                 policy_final_fc_init_scale: float = 1.0):
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
            final_fc_init_scale=policy_final_fc_init_scale)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        critic_def = critic_net.DoubleCritic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))
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
    
    def sparse_model_sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.copy_actor.apply_fn,
                                               self.copy_actor.params, observations,
                                               temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
    

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1

        new_rng, new_actor, new_critic, new_target_critic, new_temp, new_critic_grad, new_actor_grad, info = _update_jit(
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy,
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
        # else:
        #     actor_grad_info = calculate(new_actor_grad, self.actor_grad, is_actor=True)
        #     critic_grad_info= calculate(critic_grad_info, self.critic_grad, is_actor=False)
        #     self.critic_grad = new_critic_grad
        #     self.actor_grad = new_actor_grad
        
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
        
    
        
    
    def update_instant_sparsity(self, env_name, step, sparsity, pruner):
        # self.save_networks(env_name, additional_info=f"step_{step}")
        
        additional_info = 'sparsity_' + str(sparsity) + '_step_' + str(step)
        self.copy_actor = self.copy_actor.replace(step=self.actor.step, 
                                params=self.actor.params, 
                                opt_state=self.actor.opt_state)
        self.copy_critic = self.copy_critic.replace(step=self.critic.step,
                                params=self.critic.params,
                                opt_state=self.critic.opt_state)
        self.copy_target_critic = self.copy_target_critic.replace(step=self.target_critic.step,
                                        params=self.copy_target_critic.params,
                                        opt_state=self.copy_target_critic.opt_state)
        
        pruned_actor_params, _ = pruner.instant_sparsify(self.copy_actor.params)
        pruned_critic_params, _ = pruner.instant_sparsify(self.copy_critic.params)
        pruned_target_critic_params, _ = pruner.instant_sparsify(self.copy_target_critic.params)
        self.copy_actor = self.copy_actor.replace(params=pruned_actor_params)
        self.copy_critic = self.copy_critic.replace(params=pruned_critic_params)
        self.copy_target_critic = self.copy_target_critic.replace(params=pruned_target_critic_params)
        
        # self.save_sparse_copy_networks(env_name, additional_info=additional_info)
        
    def save_networks(self, env_name, additional_info=None, save_dir='./models'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if additional_info:
            save_dir = save_dir + '/' + str(env_name) + '_' + str(additional_info)
        else:
            save_dir = save_dir + '/' + str(env_name)
        self.actor.save(save_dir + '_actor.ckpt')
        self.critic.save(save_dir + '_critic.ckpt')
        self.target_critic.save(save_dir + '_target_critic.ckpt')
        
    def save_sparse_copy_networks(self, env_name, additional_info=None, save_dir='./models'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if additional_info:
            save_dir = save_dir + '/sparse_' + str(env_name) + '_' + str(additional_info)
        else:
            save_dir = save_dir + '/' + str(env_name)
        self.copy_actor.save(save_dir + '_actor.ckpt')
        self.copy_critic.save(save_dir + '_critic.ckpt')
        self.copy_target_critic.save(save_dir + '_target_critic.ckpt')
        
    def load_networks(self, env_name, additional_info=None, save_dir='./models'):
        if additional_info:
            save_dir = save_dir + '/' + str(env_name) + '_' + str(additional_info)
        else:
            save_dir = save_dir + '/' + str(env_name)
        self.actor = self.actor.load(save_dir + '_actor.ckpt')
        self.critic = self.critic.load(save_dir + '_critic.ckpt')
        self.target_critic = self.target_critic.load(save_dir + '_target_critic.ckpt')
        

