from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
from boop_env import BoopEnv

class SelfPlayBoopEnv(gym.Env):
    def __init__(self, opponent_model=None):
        super().__init__()
        self.env = BoopEnv()
        self.opponent_model = opponent_model
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6, 6, 5), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([2, 6, 6, 2])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.env.reset()

    def step(self, action):
        if self.env.current_player_num == 0:
            obs, reward, terminated, truncated, info = self.env.step(action)
            # Player 0 acted; reward is already from their perspective — do not invert
        else:
            # Player 1 (opponent) is about to act
            obs = self.env.observation
            retries = 0
            max_retries = 200
            while retries < max_retries:
                opponent_action, _ = self.opponent_model.predict(obs, deterministic=False)
                if self.env.is_legal(opponent_action):
                    obs, reward, terminated, truncated, info = self.env.step(opponent_action)
                    # Since player 1 acted, invert reward to reflect player 0's view
                    reward = -reward if reward != 0 else 0
                    break
                retries += 1

            if retries >= max_retries:
                # Opponent failed to act legally; player 0 wins by forfeit
                obs = self.env.observation
                reward = 1      # win for player 0
                terminated = True
                truncated = False
                info = {"reason": "opponent_invalid_action"}

        return obs, reward, terminated, truncated, info

# Opponent is just a random agent
class RandomOpponent:
    def predict(self, obs, deterministic=True):
        return SelfPlayBoopEnv().action_space.sample(), None

env = SelfPlayBoopEnv(opponent_model=RandomOpponent())
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_boop_v0")

opponent_model = PPO.load("ppo_boop_v0")
env = SelfPlayBoopEnv(opponent_model=opponent_model)
new_model = PPO("MlpPolicy", env, verbose=1)
new_model.learn(total_timesteps=100000)
new_model.save("ppo_boop_v1")

opponent_model = PPO.load("ppo_boop_v1")
env = SelfPlayBoopEnv(opponent_model=opponent_model)
new_model = PPO("MlpPolicy", env, verbose=1)
new_model.learn(total_timesteps=100000)
new_model.save("ppo_boop_v2")

opponent_model = PPO.load("ppo_boop_v2")
env = SelfPlayBoopEnv(opponent_model=opponent_model)
new_model = PPO("MlpPolicy", env, verbose=1)
new_model.learn(total_timesteps=100000)
new_model.save("ppo_boop_v3")
