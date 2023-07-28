from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent, env: gym.Env, num_episodes: int, sparse_model: bool) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    successes = None
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            if sparse_model:
                action = agent.sparse_model_sample_actions(observation, temperature=0.0)
            else:
                action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
        for k in stats.keys():
            stats[k].append(info['episode'][k])

        if 'is_success' in info:
            if successes is None:
                successes = 0.0
            successes += info['is_success']

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    return stats
