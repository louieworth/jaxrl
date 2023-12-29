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
from jaxrl.utils import make_env, Log, calculate

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'HalfCheetah-v4', 'Environment name.')
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
flags.DEFINE_string('wandb_project_name', "sparse_rl_tiny_network", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', "louis_t0", "the entity (team) of wandb's project")

flags.DEFINE_boolean('save_model', False, 'Save model during training.')
flags.DEFINE_integer('save_model_interval', int(5e4), 'Save model interval.')
flags.DEFINE_boolean('load_model', False, 'Load model during training.')
flags.DEFINE_boolean('negative_side_variace', False, 'Whether to calculate the negative side variance')
flags.DEFINE_boolean('reset_buffer', False, 'Whether to reset the buffer')
# flags.DEFINE_integer('instant_sparsity_interval', int(1e4), 'Reset interval for the model.')

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
    replay_buffer_size = kwargs.pop('replay_buffer_size')
    log = Log(Path(f'policy_distance')/f"{FLAGS.env_name}/{replay_buffer_size}_reset_buffer_{FLAGS.reset_buffer}", kwargs)
    log(f'Log dir: {log.dir}')
    
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
    
    if FLAGS.load_model:
        logging.info(f"load the model")
        # agent.load_avg_networks(env_name=FLAGS.env_name)
        agent.load_networks(env_name=FLAGS.env_name)
    
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 replay_buffer_size or FLAGS.max_steps)

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
                update_info, new_critic_grad, new_actor_grad = agent.update(batch)
                if FLAGS.track:
                    wandb.log(update_info, step=i)
        
            if i % FLAGS.log_interval == 1:
                agent.last_actor = agent.actor
                agent.last_critic = agent.critic
                agent.critic_grad = new_critic_grad
                agent.actor_grad = new_actor_grad
            # if i % FLAGS.eval_interval == 0:
            if i % FLAGS.log_interval == 0:
                eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes, sparse_model=False)
                if FLAGS.track:
                    wandb.log({'average_return': eval_stats['return']}, step=i)
                batch = replay_buffer.sample_top(k=256)
                actions = agent.sample_actions(batch.observations, temperature=0.0)
                actions = np.asarray(actions)
                distance = np.square(actions - batch.actions).mean()
                # distance = np.linalg.norm(actions - batch.actions).mean()
                log.row({'distance': distance})
            if FLAGS.reset_buffer and i % int(2e5) == 0:
                logging.info(f"reset the buffer")
                replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 replay_buffer_size or FLAGS.max_steps)
                k = 0
                while k < FLAGS.batch_size:
                    action = env.action_space.sample()
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
                    k += 1

                    

                
                
                # actor_grad_info = calculate(new_actor_grad, agent.actor_grad, is_actor=True, grad=True)
                # critic_grad_info = calculate(new_critic_grad, agent.critic_grad, grad=True)
                # actor_info = calculate(agent.actor.params, agent.last_actor.params, is_actor=True)
                # critic_info= calculate(agent.critic.params, agent.last_critic.params)
                # log.row({**actor_info,
                #         **critic_info,
                #         **actor_grad_info,
                #         **critic_grad_info,
                #         })
                 
        if FLAGS.save_model and i % FLAGS.save_model_interval == 0:
            logging.info(f"save the model")
            agent.save_networks(FLAGS.env_name, additional_info=f"step_{i}_seed{FLAGS.seed}", save_dir=f'./models/updates_per_step_{FLAGS.updates_per_step}')
            
if __name__ == '__main__':
    app.run(main)