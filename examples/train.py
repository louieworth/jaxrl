import os
import sys
import random
import time
import logging
import IPython
from pathlib import Path

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

from jaxrl.agents import (AWACLearner, DDPGLearner, REDQLearner, SACLearner,
                          SACV1Learner)
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env, calculate, Log

import jaxpruner
import ml_collections
FLAGS = flags.FLAGS



flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_string('save_dir', 'tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', int(1e4), 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('updates_per_step', 1, 'Gradient updates per step.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')

flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('track', False, 'Track experiments with Weights and Biases.')
flags.DEFINE_string('wandb_project_name', "small_buffer", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', "louis_t0", "the entity (team) of wandb's project")
flags.DEFINE_boolean('save_model', False, 'Save model during training.')
flags.DEFINE_integer('save_model_interval', int(5e4), 'Save model interval.')

###sparsity config
flags.DEFINE_string("prune_algorithm", "no_prune", "pruning algorithm")
# ('no_prune', 'magnitude', 'random', 'saliency', 'magnitude_ste', 'random_ste', 
# 'global_magnitude', 'global_saliency', 'static_sparse', 'rigl','set')
flags.DEFINE_integer("prune_update_freq", int(1e4), "update frequency")
flags.DEFINE_integer("prune_update_end_step", int(1e6), "update end step")
flags.DEFINE_integer("prune_update_start_step", int(1e4), "update start step")
flags.DEFINE_float("prune_actor_sparsity", 0.3, "sparsity")
flags.DEFINE_float("prune_critic_sparsity", 0.3, "sparsity")
flags.DEFINE_string("prune_dist_type", "erk", "distribution type")
flags.DEFINE_boolean('negative_side_variace', False, 'compute negative side variance')

flags.DEFINE_boolean('layer_normalization', False, 'Use layer normalization.')
flags.DEFINE_boolean('reset_memory', False, 'Reset memory.')
flags.DEFINE_integer('reset_memory_interval', int(2e5), 'Reset memory interval.')
flags.DEFINE_integer('replay_buffer_size', int(1e6), 'Replay buffer size.')


# config definition
config_flags.DEFINE_config_file(
    'config',
    'examples/configs/sac_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    kwargs = dict(FLAGS.config)
    algo = kwargs.pop('algo')
    run_name = f"{FLAGS.env_name}__{algo}__{FLAGS.seed}"
    if FLAGS.track:
        import wandb
        
        clean_config = {}
        clean_config['algo'] = algo
        
        clean_config['env_name'] = FLAGS.env_name
        clean_config['seed'] = FLAGS.seed
        clean_config['layer_normalization'] = FLAGS.layer_normalization
        clean_config['actor_lr']=kwargs['actor_lr']
        clean_config['critic_lr']=kwargs['critic_lr']
        
        clean_config['prune_algorithm']=FLAGS.prune_algorithm
        clean_config['prune_update_freq']=FLAGS.prune_update_freq
        clean_config['prune_update_start_step']=FLAGS.prune_update_start_step
        clean_config['prune_actor_sparsity']=FLAGS.prune_actor_sparsity
        clean_config['prune_critic_sparsity']=FLAGS.prune_critic_sparsity
        clean_config['reset_memory']=FLAGS.reset_memory
        clean_config['reset_memory_interval']=FLAGS.reset_memory_interval

        wandb.init(
            project=FLAGS.wandb_project_name,
            entity=FLAGS.wandb_entity,
            # sync_tensorboard=True,
            config=clean_config,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.config.update({"algo": algo})
    
    # log_config = {**kwargs, **{k: v for k, v in clean_config.items() if k not in kwargs}}
    log = Log(Path(f'magnitude_wsr_last_{FLAGS.prune_actor_sparsity}')/FLAGS.env_name, kwargs)
    log(f'Log dir: {log.dir}')
    
        
    sparsity_config = ml_collections.ConfigDict()
    sparsity_config.algorithm = FLAGS.prune_algorithm
    sparsity_config.update_freq = FLAGS.prune_update_freq
    sparsity_config.update_end_step = FLAGS.prune_update_end_step
    sparsity_config.update_start_step = FLAGS.prune_update_start_step
    sparsity_config.sparsity = FLAGS.prune_critic_sparsity
    sparsity_config.dist_type = FLAGS.prune_dist_type
    
    actor_pruner = jaxpruner.create_updater_from_config(sparsity_config)
    sparsity_config.sparsity = FLAGS.prune_critic_sparsity
    critic_pruner = jaxpruner.create_updater_from_config(sparsity_config)
    kwargs['layer_normalization'] = FLAGS.layer_normalization
    kwargs['actor_pruner'] = actor_pruner
    kwargs['critic_pruner'] = critic_pruner
    logging.info(f"pruner_algorithm: {FLAGS.prune_algorithm}")
    logging.info(f"algo: {algo}")

    # summary_writer = SummaryWriter(
    #     os.path.join(FLAGS.save_dir, run_name))

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env = make_env(FLAGS.env_name, FLAGS.seed, video_train_folder)
    eval_env = make_env(FLAGS.env_name, FLAGS.seed + 42, video_eval_folder)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    if algo == 'sac':
        agent = SACLearner(FLAGS.seed,
                           env.observation_space.sample()[np.newaxis],
                           env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'redq':
        agent = REDQLearner(FLAGS.seed,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis],
                            policy_update_delay=FLAGS.updates_per_step,
                            **kwargs)
    elif algo == 'sac_v1':
        agent = SACV1Learner(FLAGS.seed,
                             env.observation_space.sample()[np.newaxis],
                             env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'awac':
        agent = AWACLearner(FLAGS.seed,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'ddpg':
        agent = DDPGLearner(FLAGS.seed,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis], **kwargs)
    else:
        raise NotImplementedError()
    
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 FLAGS.replay_buffer_size or FLAGS.max_steps)

    observation, done = env.reset(), False
    use_init_agent = True
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask, float(done),
                             next_observation)
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            
        if i >= FLAGS.start_training:
            for _ in range(FLAGS.updates_per_step):
                batch = replay_buffer.sample(FLAGS.batch_size)
                update_info, new_critic_grad, new_actor_grad = agent.update(batch)

            if FLAGS.track and i % FLAGS.log_interval == 0:
                wandb.log(update_info, step=i)

            if i % FLAGS.eval_interval == 1:
                    agent.last_actor = agent.actor
                    agent.last_critic = agent.critic
                    agent.critic_grad = new_critic_grad
                    agent.actor_grad = new_actor_grad
                    use_init_agent = False
                    
            if i % FLAGS.eval_interval == 0:
                eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)
                if FLAGS.track:
                    wandb.log({'average_return': eval_stats['return']}, step=i)
                # log.row({'average_return': eval_stats['return']})
                else:
                    actor_grad_info = calculate(new_actor_grad, agent.actor_grad, is_actor=True, grad=True)
                    critic_grad_info = calculate(new_critic_grad, agent.critic_grad, grad=True)
                    actor_info = calculate(agent.actor.params, agent.last_actor.params, is_actor=True)
                    critic_info= calculate(agent.critic.params, agent.last_critic.params)
                    log.row({**actor_info,
                            **critic_info,
                            **actor_grad_info,
                            **critic_grad_info,
                            })

            if FLAGS.reset_memory and i % FLAGS.reset_memory_interval == 0:
                print('------------reset memory-----------')
                replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                        FLAGS.replay_buffer_size or FLAGS.max_steps)
                
                add_total_step = i + FLAGS.start_training
                add_current_step = i
                observation, done = env.reset(), False
                while add_current_step < add_total_step:
                    action = agent.sample_actions(observation)
                    next_observation, reward, done, info = env.step(action)

                    if not done or 'TimeLimit.truncated' in info:
                        mask = 1.0
                    else:
                        mask = 0.0

                    replay_buffer.insert(observation, action, reward, mask, float(done),
                                        next_observation)
                    observation = next_observation
                    if done:
                        observation, done = env.reset(), False
                    add_current_step += 1
                                
        if FLAGS.save_model and i % FLAGS.save_model_interval == 0:
            logging.info(f"save the model")
            agent.save_networks(FLAGS.env_name, additional_info=f"step_{i}_seed{FLAGS.seed}", save_dir=f'./models/updates_per_step_{FLAGS.updates_per_step}')
            
            

if __name__ == '__main__':
    app.run(main)