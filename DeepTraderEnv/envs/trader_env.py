import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
from random import randint

class trader_env(gym.Env):
  #metadata = {'render.modes': ['human']}

  def __init__(self, candles=50, start_time="06:00:00", end_time="19:00:00"):
    self.df = pd.read_csv("/Users/Drew/Documents/Forex/Candlesticks/GBPUSD_data.csv")
    self.num_candles = candles
    self.action_space = spaces.Discrete(3)
    self.observation_space = spaces.Box(low=0, high=1,shape=(candles*4,))
    self.df['day_time'] = self.df['time'].str.rsplit(' ', expand=True)[1]
    self.start_time = start_time
    self.end_time = end_time
    self.num_episodes = sum(self.df['day_time'] == self.start_time) - 1
    self.n = np.where(self.df['day_time'] == self.start_time)[0][randint(0, self.num_episodes - 1)]
    self.state = self.get_state()
    self.position = 0
    self.last_position = 0
    self.trade_period = 0

  def _step(self, action):
    reward = self.collect_reward()
    self.last_position = self.position
    self.position = action - 1
    self.state = self.get_state()
    self.n += 1
    self.trade_period += 1
    return self.state, reward, self.is_over(), {}

  def _reset(self):
    self.n = np.where(self.df['day_time'] == self.start_time)[0][randint(0, self.num_episodes - 1)]
    self.state = self.get_state()
    self.position = 0
    self.last_position = 0
    self.trade_period = 0
    return self.state

  def _render(self, mode='human', close=False):
    return

  def get_state(self):  # scaled between 0 and 1
    result = self.df.iloc[self.n - self.num_candles:self.n][['open', 'close', 'high', 'low']].values.ravel()
    if len(result) < 2:
        print self.n
    old_min = min(result)
    old_max = max(result)
    old_range = old_max-old_min
    result = (result-old_min)/old_range
    return result

  def collect_reward(self):
    reward = self.position * (self.df.iloc[self.n]['close'] - self.df.iloc[self.n - 1]['close']) * 10000
    if self.position != self.last_position:
       reward -= 1.5
    return reward

  def is_over(self):
    return self.df.iloc[self.n].get_value('day_time') == self.end_time