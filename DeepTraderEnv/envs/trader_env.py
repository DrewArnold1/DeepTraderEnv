import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
from random import randint

class trader_env(gym.Env):
  #metadata = {'render.modes': ['human']}

  def __init__(self, candles=50, start_time="06:00:00", end_time="19:00:00"):
    df = pd.read_csv("/Users/Drew/Documents/Forex/Candlesticks/GBPUSD_data.csv")
    self.num_candles = candles
    self.df['day_time'] = self.df['time'].str.rsplit(' ', expand=True)[1]
    self.start_time = start_time
    self.end_time = end_time
    self.num_episodes = sum(self.df['day_time'] == self.start_time) - 1
    self.n = np.where(self.df['day_time'] == self.start_time)[0][randint(0, self.num_episodes - 1)]
    self.state = self.get_state(self)
    self.position = 0
    self.last_position = 0
    self.trade_period = 0

  def _step(self, action):
    reward = self.collect_reward(self, action)
    self.last_position = self.position
    self.position = action - 1
    self.state = self.get_state(self)
    self.n += 1
    self.trade_period += 1
    return self.state, reward, self.is_over(), {}

  def _reset(self):
    self.n = randint(1, len(self.df.index) - 200)
    self.state = self.get_state(self)
    self.position = 0
    self.last_position = 0
    self.trade_period = 0
    return self.state

  def _render(self, mode='human', close=False):
    return

  def get_state(self):  # scaled so final close = 0
    result = self.df.iloc[self.n - self.num_candles:self.n][['open', 'close', 'high', 'low']].values.ravel()
    return result - result[len(result) - 3]

  def collect_reward(self, action):
    reward = self.position * (self.df.iloc[self.n]['close'] - self.df.iloc[self.n - 1]['close']) * 100
    # if self.position != self.last_position:
    #    reward -= 1.5
    return reward

  def is_over(self):
    return self.df.iloc[self.n].get_value('day_time') == self.end_time