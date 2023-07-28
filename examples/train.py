import os
import random
import functools
import sys
from pathlib import Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

import numpy as np
import tqdm
import logging
from absl import app, flags
from ml_collections import config_flags
import jaxpruner

from jaxrl.agents import (AWACLearner, DDPGLearner, REDQLearner, SACLearner,
                          SACV1Learner)
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env, Log, calculate_scores

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
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
flags.DEFINE_string('wandb_project_name', "sparse_rl", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', "louis_t0", "the entity (team) of wandb's project")

flags.DEFINE_boolean('save_model', False, 'Save model during training.')
flags.DEFINE_integer('save_model_interval', int(1e6), 'Save model interval.')
flags.DEFINE_boolean('load_model', False, 'Load model during training.')
flags.DEFINE_boolean('negative_side_variace', True, 'Whether to calculate the negative side variance')
# flags.DEFINE_integer('instant_sparsity_interval', int(1e4), 'Reset interval for the model.')

###sparsity config
flags.DEFINE_float("init_sparsity", 0.1, "initial sparsity")
flags.DEFINE_float("target_sparsity", 0.3, "target sparsity")

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
        clean_config['actor_lr']=kwargs['actor_lr']
        clean_config['critic_lr']=kwargs['critic_lr']

        wandb.init(
            project=FLAGS.wandb_project_name,
            entity=FLAGS.wandb_entity,
            config=clean_config,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.config.update({"algo": algo})
        
    
    log = Log(Path('negative_side_variance')/FLAGS.env_name, kwargs)
    log(f'Log dir: {log.dir}')
    
    # create the pruner
    sparsity_distribution = functools.partial(
        jaxpruner.sparsity_distributions.uniform, sparsity=FLAGS.init_sparsity)
    pruner = jaxpruner.MagnitudePruning(sparsity_distribution_fn=sparsity_distribution)
    # Gradient based pruning
    # pruner = jaxpruner.SaliencyPruning(sparsity_distribution_fn=sparsity_distribution)

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


    replay_buffer_size = kwargs.pop('replay_buffer_size')
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
    
    if FLAGS.load_model:
        logging.info(f"load the model")
        # agent.load_avg_networks(env_name=FLAGS.env_name)
        agent.load_networks(env_name=FLAGS.env_name)
    
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 replay_buffer_size or FLAGS.max_steps)

    sparsity_distribution = functools.partial(
                jaxpruner.sparsity_distributions.uniform, sparsity=FLAGS.init_sparsity)
    pruner = jaxpruner.MagnitudePruning(sparsity_distribution_fn=sparsity_distribution)
    observation, done = env.reset(), False
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
                update_info = agent.update(batch)
        
        if i % FLAGS.eval_interval == 0:
            sparsity = FLAGS.init_sparsity + (FLAGS.target_sparsity - FLAGS.init_sparsity) * i / FLAGS.max_steps
            agent.update_instant_sparsity(FLAGS.env_name, i, sparsity, pruner)
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes, sparse_model=False)
            sparse_eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes, sparse_model=True)
            actor_sparsity_level = jaxpruner.summarize_sparsity(agent.copy_actor.params, only_total_sparsity=True)
            critic_sparsity_level = jaxpruner.summarize_sparsity(agent.copy_critic.params, only_total_sparsity=True)
            # wandb.log({'average_return': eval_stats['return'],
            #             'sparse_average_return': sparse_eval_stats['return'],
            #             'actor_sparsity_level': actor_sparsity_level,
            #             'critic_sparsity_level': critic_sparsity_level}, step=i)
            
            # log.row({'normalized_return': eval_stats['return']})
            if FLAGS.negative_side_variace:
                actor_layer_0, actor_layer_1, actor_layer_2, actor_layer_3, actor_total = calculate_scores(agent.actor.params, negative_bias=True, is_actor=True)
                critic_layer_0, critic_layer_1, critic_layer_2, critic_total = calculate_scores(agent.critic.params, negative_bias=True, is_actor=False)
                sparse_actor_layer_0, sparse_actor_layer_1, sparse_actor_layer_2, sparse_actor_layer_3, sparse_actor_total = calculate_scores(agent.copy_actor.params, negative_bias=True, is_actor=True)
                sparse_critic_layer_0, sparse_critic_layer_1, sparse_critic_layer_2, sparse_critic_total = calculate_scores(agent.copy_critic.params, negative_bias=True, is_actor=False)
                log.row({'actor_layer_0': actor_layer_0,
                         'sparse_actor_layer_0': sparse_actor_layer_0,
                         'actor_layer_1': actor_layer_1,
                         'sparse_actor_layer_1': sparse_actor_layer_1,
                         'actor_layer_2': actor_layer_2,
                         'sparse_actor_layer_2': sparse_actor_layer_2,
                         'actor_layer_3': actor_layer_3,
                         'sparse_actor_layer_3': sparse_actor_layer_3,
                         'critic_layer_0': critic_layer_0,
                         'sparse_critic_layer_0': sparse_critic_layer_0,
                         'critic_layer_1': critic_layer_1,
                         'sparse_critic_layer_1': sparse_critic_layer_1,
                         'critic_layer_2': critic_layer_2,
                         'sparse_critic_layer_2': sparse_critic_layer_2,
                         'actor_total': actor_total,
                         'sparse_actor_total': sparse_actor_total,
                         'critic_total': critic_total,
                         'sparse_critic_total': sparse_critic_total,
                         'average_return': eval_stats['return'],
                         'sparse_average_return': sparse_eval_stats['return'],
                         'sparsity': sparsity
                         })
                
        if FLAGS.save_model and i % FLAGS.save_model_interval == 0:
            logging.info(f"save the model")
            agent.save_networks(FLAGS.env_name, additional_info=f"step_{i}")
            
if __name__ == '__main__':
    app.run(main)