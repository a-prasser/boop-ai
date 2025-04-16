import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from tictactoe_env import TicTacToeEnv

# Register custom env (only needed if you want to use gym.make)
gym.envs.registration.register(
    id='TicTacToe-v0',
    entry_point='tictactoe_env:TicTacToeEnv',
)

env = make_vec_env('TicTacToe-v0', n_envs=1)

model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=10000, exploration_fraction=0.2)
model.learn(total_timesteps=50000)

model.save("tictactoe_dqn")
