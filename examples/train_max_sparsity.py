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
import jax.numpy as jnp
from absl import app, flags
from ml_collections import config_flags
import jaxpruner

from jaxrl.agents import (AWACLearner, DDPGLearner, REDQLearner, SACLearner,
                          SACV1Learner)
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env, Log, calculate_scores

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'HalfCheetah-v4', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('updates_per_step', 1, 'Gradient updates per step.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('track', True, 'Track experiments with Weights and Biases.')
flags.DEFINE_string('wandb_project_name', "sparse_rl", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', "louis_t0", "the entity (team) of wandb's project")
flags.DEFINE_boolean('negative_weight', True, 'Whether to calculate the negative weight')
flags.DEFINE_integer('negative_weight_interval', int(2e4), 'Interval to calculate the negative weight')

flags.DEFINE_boolean('save_model', True, 'Save model during training.')
flags.DEFINE_integer('save_model_interval', int(1e6), 'Save model interval.')
flags.DEFINE_boolean('load_model', False, 'Load model during training.')
flags.DEFINE_integer('instant_sparsity_interval', int(1e4), 'Reset interval for the model.')

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
        
    
    log = Log(Path('max_performance_gap_0.1')/FLAGS.env_name, kwargs)
    log(f'Log dir: {log.dir}')
    
    # create the pruner

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
    
    last_step = step = int(1e5)

    
    while step < FLAGS.max_steps:
        # agent.load_avg_networks(env_name=FLAGS.env_name)
        last_sparsity = sparsity = 0
        last_sparse_return = last_return = 0
        
        while sparsity < 1:
            agent.load_networks(env_name=FLAGS.env_name, additional_info=f"step_{step}")
            
            sparsity_distribution = functools.partial(
            jaxpruner.sparsity_distributions.uniform, sparsity=sparsity)
            pruner = jaxpruner.MagnitudePruning(sparsity_distribution_fn=sparsity_distribution)
        
            agent.update_instant_sparsity(FLAGS.env_name, step, sparsity, pruner)
            eval_stats = evaluate(agent, eval_env, 50, sparse_model=False)
            sparse_eval_stats = evaluate(agent, eval_env, 50, sparse_model=True)
            performance_gap_ratio =  jnp.abs(eval_stats['return'] - sparse_eval_stats['return']) /  eval_stats['return']
             
            if performance_gap_ratio > 0.1:
                log.row({'step': last_step, 
                        'sparsity': last_sparsity, 
                        'return': last_return,
                        'sparse_return': last_sparse_return,
                        'performance_gap_ratio': last_performance_gap_ratio})
                log(f"WARING: step: {step}, sparsity: {sparsity}, return: {eval_stats['return']}, sparse_return: {sparse_eval_stats['return']}, performance_gap_ratio: {performance_gap_ratio}")
                sparsity = 0
                break
            else:
                sparsity += 0.01
                last_step = step
                last_sparsity = sparsity
                last_return = eval_stats['return']
                last_sparse_return = sparse_eval_stats['return']
                last_performance_gap_ratio = performance_gap_ratio
                log(f"step: {step}, sparsity: {sparsity}, return: {eval_stats['return']}, sparse_return: {sparse_eval_stats['return']}, performance_gap_ratio: {performance_gap_ratio}")
        step += int(1e5)
        

            
if __name__ == '__main__':
    app.run(main)