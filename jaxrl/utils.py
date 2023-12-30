from typing import Optional
import collections
import csv
import json
import random
import string
import sys

from sqlite3 import DatabaseError
from typing import Optional
from datetime import datetime
from pathlib import Path
import numpy as np

import jax
import jax.numpy as jnp

import gym
from gym.wrappers import RescaleAction
from gym.wrappers.pixel_observation import PixelObservationWrapper

from jaxrl import wrappers


def make_env(env_name: str,
             seed: int,
             save_folder: Optional[str] = None,
             add_episode_monitor: bool = True,
             action_repeat: int = 1,
             frame_stack: int = 1,
             from_pixels: bool = False,
             pixels_only: bool = True,
             image_size: int = 84,
             sticky: bool = False,
             gray_scale: bool = False,
             flatten: bool = True) -> gym.Env:
    # Check if the env is in gym.
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    if env_name in env_ids:
        env = gym.make(env_name)
    else:
        domain_name, task_name = env_name.split('-')
        env = wrappers.DMCEnv(domain_name=domain_name,
                              task_name=task_name,
                              task_kwargs={'random': seed})

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if add_episode_monitor:
        env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    env = RescaleAction(env, -1.0, 1.0)

    if save_folder is not None:
        env = gym.wrappers.RecordVideo(env, save_folder)

    if from_pixels:
        if env_name in env_ids:
            camera_id = 0
        else:
            camera_id = 2 if domain_name == 'quadruped' else 0
        env = PixelObservationWrapper(env,
                                      pixels_only=pixels_only,
                                      render_kwargs={
                                          'pixels': {
                                              'height': image_size,
                                              'width': image_size,
                                              'camera_id': camera_id
                                          }
                                      })
        env = wrappers.TakeKey(env, take_key='pixels')
        if gray_scale:
            env = wrappers.RGB2Gray(env)
    else:
        env = wrappers.SinglePrecision(env)

    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    if sticky:
        env = wrappers.StickyActionEnv(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env

def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{rand_str}'

class Log:
    def __init__(self, root_log_dir, cfg_dict,
                 txt_filename='log.txt',
                 csv_filename='progress.csv',
                 cfg_filename='config.json',
                 flush=True):
        self.dir = Path(root_log_dir)/_gen_dir_name()
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir/txt_filename, 'w')
        self.csv_file = None
        (self.dir/cfg_filename).write_text(json.dumps(cfg_dict))
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        self.cfg_filename = cfg_filename
        self.flush = flush

    def write(self, message, end='\n'):
        now_str = datetime.now().strftime('%H:%M:%S')
        message = f'[{now_str}] ' + message
        for f in [sys.stdout, self.txt_file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict):
        if self.csv_file is None:
            self.csv_file = open(self.dir/self.csv_filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, list(dict.keys()))
            self.csv_writer.writeheader()

        self(str(dict))
        self.csv_writer.writerow(dict)
        if self.flush:
            self.csv_file.flush()

    def close(self):
        self.txt_file.close()
        if self.csv_file is not None:
            self.csv_file.close()
                     
def calculate_wdi(mag, last_mag, is_actor=False):
    if is_actor:
        N = mag.shape[0] * mag.shape[1]
    else:
        N = mag.shape[0] * mag.shape[1] * mag.shape[2]
    num_neurons_smaller_than_last_mag = jnp.where(mag < last_mag, 1, 0)
    A = num_neurons_smaller_than_last_mag * mag
    B = num_neurons_smaller_than_last_mag * last_mag
    overlap_co = jnp.sum(jnp.minimum(A, B)) / (jnp.sum(jnp.maximum(A, B)) + 1e-8)
    
    return num_neurons_smaller_than_last_mag.mean(), overlap_co

def calculate(params, last_params, is_actor=False, grad=False):
    param_magnitudes = jax.tree_map(lambda p: jnp.abs(p), params)
    last_layer_params_magnitudes = jax.tree_map(lambda p: jnp.abs(p), last_params)
    if is_actor:
        ratio_neurons_smaller_than_last_mags = []
        overlap_cos = []
        # Iterate over layers and compute mean and count of kernels smaller than mean
        for layer_idx, (mag, last_mag) in enumerate(zip(jax.tree_util.tree_leaves(param_magnitudes), jax.tree_util.tree_leaves(last_layer_params_magnitudes))):
            if len(mag.shape) > 1:
                # number of nuerons smaller than last magnitude
                ratio_neurons_smaller_than_last_mag, overlap_co = calculate_wdi(mag, last_mag, is_actor=True)
                ratio_neurons_smaller_than_last_mags.append(ratio_neurons_smaller_than_last_mag)
                overlap_cos.append(overlap_co)
        if grad: 
            return {'actor_grad_wsr': round(np.mean(ratio_neurons_smaller_than_last_mags) * 100, 2),
                    'actor_grad_overlap_co': round(np.mean(overlap_cos)* 100, 2)}
        else: 
            return {'actor_weight_wsr': round(np.mean(ratio_neurons_smaller_than_last_mags) * 100, 2), 
                    'actor_weight_overlap_co': round(np.mean(overlap_cos)* 100, 2)}
            
    else:
        ratio_neurons_smaller_than_last_mags = []
        overlap_cos = []
        # Iterate over layers and compute mean and count of kernels smaller than mean
        for layer_idx, (mag, last_mag) in enumerate(zip(jax.tree_util.tree_leaves(param_magnitudes), jax.tree_util.tree_leaves(last_layer_params_magnitudes))):
            if len(mag.shape) > 2:
                ratio_neurons_smaller_than_last_mag, overlap_co = calculate_wdi(mag, last_mag)
                ratio_neurons_smaller_than_last_mags.append(ratio_neurons_smaller_than_last_mag)
                overlap_cos.append(overlap_co) 

        if grad: 
            return {'critic_grad_wsr': round(np.mean(ratio_neurons_smaller_than_last_mags) * 100, 2),
                    'critic_grad_overlap_co': round(np.mean(overlap_cos)* 100, 2)
                    }
        else: 
            return {'critic_weight_wsr': round(np.mean(ratio_neurons_smaller_than_last_mags) * 100, 2), 
                    'critic_weight_overlap_co': round(np.mean(overlap_cos)* 100, 2)}
            

