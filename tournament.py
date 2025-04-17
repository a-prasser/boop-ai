from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
from boop_env import BoopEnv

def play_match(model0, model1, env, verbose=False):
    env.reset()
    done = False
    while not done:
        obs = env.observation
        current = env.current_player_num

        # Choose model for current player
        model = model0 if current == 0 else model1

        max_tries = 300
        tries = 0
        while tries < max_tries:
            action, _ = model.predict(obs, deterministic=False)
            if env.is_legal(action):
                obs, reward, done, truncated, info = env.step(action)
                break
            tries+=1
        if tries >= max_tries:
            obs, reward, done, truncated, info = env.step(action)
            
        if verbose:
            print(f"Player {current} moved. Reward: {reward}, Done: {done}")

    winner = None
    if reward == 1:
        winner = env.current_player_num
    elif reward == -1:
        winner = 1 - env.current_player_num

    return winner, reward

models = [PPO.load(f"ppo_boop_v{i}") for i in range(4)]

matches = 20
wins = {f'model_{i}':0 for i in range(4)}

for i in range(len(models)):
    for j in range(i + 1, len(models)):
        for match in range(matches):
            env = BoopEnv()
            winner, _ = play_match(models[i], models[j], env, verbose=False)
            print(f"Match: model_{i} vs model_{j} → Winner: {winner}")
            if winner == 0:
                wins[f'model_{i}'] += 1
            else:
                wins[f'model_{i}'] += 1
print(wins)