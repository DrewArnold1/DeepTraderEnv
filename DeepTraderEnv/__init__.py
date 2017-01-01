from gym.envs.registration import register

register(
    id='DeepTraderEnv-v0',
    entry_point='DeepTraderEnv.envs:DeepTraderEnv',
)
