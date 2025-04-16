# tictactoe_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=2, shape=(9,), dtype=np.int8)
        self.action_space = spaces.Discrete(9)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=np.int8)  # 0=empty, 1=agent, 2=opponent
        self.done = False
        self.current_player = 1
        return self.board.copy(), {}

    def step(self, action):
        if self.done or self.board[action] != 0:
            return self.board.copy(), -1, True, False, {}

        self.board[action] = 1  # Agent move
        if self.check_winner(1):
            return self.board.copy(), 1.0, True, False, {}

        if 0 not in self.board:
            return self.board.copy(), 0.5, True, False, {}

        # Opponent move (random)
        available = np.where(self.board == 0)[0]
        if len(available) > 0:
            opponent_action = np.random.choice(available)
            self.board[opponent_action] = 2
            if self.check_winner(2):
                return self.board.copy(), -1.0, True, False, {}

        if 0 not in self.board:
            return self.board.copy(), 0.5, True, False, {}

        return self.board.copy(), 0.0, False, False, {}

    def check_winner(self, player):
        b = self.board.reshape((3, 3))
        return any(
            np.all(b[i, :] == player) or
            np.all(b[:, i] == player) for i in range(3)
        ) or np.all(np.diag(b) == player) or np.all(np.diag(np.fliplr(b)) == player)

    def render(self):
        symbols = {0: '.', 1: 'X', 2: 'O'}
        print("\n".join(" ".join(symbols[c] for c in self.board[i*3:(i+1)*3]) for i in range(3)))
        print()
