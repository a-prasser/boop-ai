from boop_env import BoopEnv
from stable_baselines3 import PPO

env = BoopEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("boop_ppo")
