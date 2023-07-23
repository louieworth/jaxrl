import os
import random
import time
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

import numpy as np
import tqdm
import logging
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from jaxrl.agents import (AWACLearner, DDPGLearner, REDQLearner, SACLearner,
                          SACV1Learner)
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env

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

flags.DEFINE_boolean('save_model', True, 'Save model during training.')
flags.DEFINE_integer('save_model_interval', int(1e6), 'Save model interval.')
flags.DEFINE_boolean('load_model', False, 'Load model during training.')
config_flags.DEFINE_config_file(
    'config',
    'examples/configs/sac_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    kwargs = dict(FLAGS.config)
    algo = kwargs.pop('algo')
    run_name = f"layer_norm_{FLAGS.env_name}__{algo}__{FLAGS.seed}"
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
            # sync_tensorboard=True,
            config=clean_config,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.config.update({"algo": algo})
        
    

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

            if i % FLAGS.log_interval == 0:
                wandb.log(update_info, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)
            wandb.log({'average_return': eval_stats['return']}, step=i)
            
        if FLAGS.save_model and i % FLAGS.save_model_interval == 0:
            logging.info(f"save the model")
            agent.save_networks(FLAGS.env_name)


            # agent.load_saved_model(actor_name='actor_0.npy', critic_name='critic_0.npy', target_critic_name='target_critic_0.npy')

            # for k, v in eval_stats.items():
            #     summary_writer.add_scalar(f'evaluation/average_{k}s', v,
            #                               info['total']['timesteps'])
            # summary_writer.flush()

            # eval_returns.append(
            #     (info['total']['timesteps'], eval_stats['return']))
            # np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
            #            eval_returns,
            #            fmt=['%d', '%.1f'])
            
if __name__ == '__main__':
    app.run(main)