
from stable_baselines3 import PPO
import os
import random
import numpy as np
from boop_env import BoopEnv
from boop_selfplay import SelfPlayBoopEnv, RandomOpponent

def play_match(model_a, model_b, games=5):
    a_wins = 0
    b_wins = 0
    for _ in range(games):
        env = BoopEnv()
        done = False
        env.reset()
        while not done:
            current = env.current_player_num
            model = model_a if current == 0 else model_b
            action, _ = model.predict(env.observation, deterministic=False)
            try:
                _, reward, done, _, _ = env.step(action)
            except:
                reward = 1 if current == 1 else -1
                done = True
        winner = env.current_player_num if reward == 1 else 1 - env.current_player_num
        if winner == 0:
            a_wins += 1
        else:
            b_wins += 1
    return a_wins, b_wins

def train_generation(gen_id, n_agents=10, parents=None, timesteps=100_000):
    os.makedirs(f"ppo_boop/gen_{gen_id:02}", exist_ok=True)
    for i in range(n_agents):
        if parents:
            opponent_path = random.choice(parents)
            opponent = PPO.load(opponent_path)
        else:
            opponent = RandomOpponent()
        env = SelfPlayBoopEnv(opponent_model=opponent)
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=timesteps)
        model.save(f"ppo_boop/gen_{gen_id:02}/v{i}")

def tournament(gen_id, n_agents=10, games_per_match=5, number_best_agents=3):
    models = [PPO.load(f"ppo_boop/gen_{gen_id:02}/v{i}") for i in range(n_agents)]
    scores = np.zeros(n_agents)
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            win_i, win_j = play_match(models[i], models[j], games=games_per_match)
            scores[i] += win_i
            scores[j] += win_j
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    print(ranked)
    top_paths = [f"ppo_boop/gen_{gen_id:02}/v{idx}" for idx, _ in ranked[:number_best_agents]]
    return top_paths

def evolve(generations=5, agents_per_gen=10, agents_to_keep=3, timesteps=100000):
    top_parents = None
    for gen in range(generations):
        print(f"\n🧬 Training Generation {gen}")
        train_generation(gen, n_agents=agents_per_gen, parents=top_parents, timesteps=timesteps if gen==0 else timesteps/4)
        print(f"🏆 Running Tournament for Generation {gen}")
        top_parents = tournament(gen, n_agents=agents_per_gen, games_per_match=10, number_best_agents=agents_to_keep)

if __name__ == "__main__":
    evolve(generations=5, agents_per_gen=8, agents_to_keep=3, timesteps=800000)

