import numpy as np
import gym
import DeepTraderEnv
import os
import pandas as pd

def running_mean(x, N):
    x = np.array(x, dtype='float64')
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

def score_from_merged(episode_lengths, episode_rewards, episode_types, timestamps, initial_reset_timestamp, trials, reward_threshold):
    """Method to calculate the score from merged monitor files. Scores
    only a single environment; mostly legacy.
    """
    if episode_types is not None:
        # Select only the training episodes
        episode_types = np.array(episode_types)
        (t_idx,) = np.where(episode_types == 't')
        episode_lengths = np.array(episode_lengths)[t_idx]
        episode_rewards = np.array(episode_rewards)[t_idx]
        timestamps = np.array(timestamps)[t_idx]

    # Make sure everything is a float -- no pesky ints.
    episode_rewards = np.array(episode_rewards, dtype='float64')

    episode_t_value = timestep_t_value = mean = error = None
    seconds_to_solve = seconds_in_total = None

    if len(timestamps) > 0:
        # This is: time from the first reset to the end of the last episode
        seconds_in_total = timestamps[-1] - initial_reset_timestamp
    if len(episode_rewards) >= trials:
        means = running_mean(episode_rewards, trials)
        if reward_threshold is not None:
            # Compute t-value by finding the first index at or above
            # the threshold. It comes out as a singleton tuple.
            (indexes_above_threshold, ) = np.where(means >= reward_threshold)
            if len(indexes_above_threshold) > 0:
                # Grab the first episode index that is above the threshold value
                episode_t_value = indexes_above_threshold[0]

                # Find timestep corresponding to this episode
                cumulative_timesteps = np.cumsum(np.insert(episode_lengths, 0, 0))
                # Convert that into timesteps
                timestep_t_value = cumulative_timesteps[episode_t_value]
                # This is: time from the first reset to the end of the first solving episode
                seconds_to_solve = timestamps[episode_t_value] - initial_reset_timestamp

        # Find the window with the best mean
        best_idx = np.argmax(means)
        best_rewards = episode_rewards[best_idx:best_idx+trials]
        mean = np.mean(best_rewards)
        if trials == 1: # avoid NaN
            error = 0.
        else:
            error = np.std(best_rewards) / (np.sqrt(trials) - 1)

    return {
        'episode_t_value': episode_t_value,
        'timestep_t_value': timestep_t_value,
        'mean': mean,
        'error': error,
        'number_episodes': len(episode_rewards),
        'number_timesteps': sum(episode_lengths),
        'seconds_to_solve': seconds_to_solve,
        'seconds_in_total': seconds_in_total,
    }

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
