import numpy as np
import gym
import DeepTraderEnv
import os
import pandas as pd

def running_mean(x, N):
    x = np.array(x, dtype='float64')
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

def score_from_local(directory):
    """Calculate score from a local results directory"""
    results = gym.monitoring.load_results(directory)
    # No scores yet saved
    if results is None:
        return None

    episode_lengths = results['episode_lengths']
    episode_rewards = results['episode_rewards']
    episode_types = results['episode_types']
    timestamps = results['timestamps']
    initial_reset_timestamp = results['initial_reset_timestamp']
    spec = gym.spec(results['env_info']['env_id'])


    return {
        'mean': np.mean(episode_rewards),
        'mean last 100': np.mean(episode_rewards[-100:]),
        'episodes': len(episode_rewards),
        'mean last 1000': np.mean(episode_rewards[-1000:])
    }

    # return score_from_merged(episode_lengths, episode_rewards, episode_types, timestamps, initial_reset_timestamp, spec.trials, spec.reward_threshold)

scores = pd.DataFrame()

for fn in os.listdir('../agents/tmp/'):
    score = score_from_local('../agents/tmp/' + fn)
    score = pd.DataFrame.from_records(score, index=[fn])
    scores = scores.append(score)

print scores
