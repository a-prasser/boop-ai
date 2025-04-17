from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from stable_baselines3 import PPO, DQN
import numpy as np
import os
import uuid

from boop_env import BoopEnv

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()            # Logs to the terminal
    ]
)
logger = logging.getLogger(__name__)

# === FastAPI setup ===
app = FastAPI()

# === Models ===
games = {
    "tictactoe": {'env': None,
                  'model': "tictactoe_dqn" },
    "boop": {'env': BoopEnv,
             'model': PPO.load("ppo_boop_v3") } # DQN.load("boop_dqn") },
    }

# === Session store ===
game_sessions = {}

# === Init new game ===
class NewGameRequest(BaseModel):
    players: list[str]

@app.post("/api/games/{game}/new")
def new_game(game: str, config: NewGameRequest):
    if game not in games:
        raise HTTPException(status_code=404, detail=f"Game '{game}' not supported")

    game_id = str(uuid.uuid4())
    env = games[game]['env']()
    env.reset()

    game_sessions[game_id] = {
        "env": env,
        "players": config.players
    }
    state = env.get_state()
    return {"game_id": game_id, "state": state, "status": "Game started"}

# === Game move ===
class MoveRequest(BaseModel):
    game_id: str
    action: list[int]

@app.post("/api/games/{game}/move")
def make_move(game: str, request: MoveRequest):
    game_id = request.game_id
    action = request.action

    if game_id not in game_sessions:
        raise HTTPException(status_code=400, detail="Invalid game ID")

    session = game_sessions[game_id]
    env = session["env"]
    players = session["players"]

    # Validate the action
    if tuple(action) not in env.legal_actions():
        return {
            "state": env.get_state(),
            "status": "Invalid action! Try again.",
            "game_over": False
        }

    # Player move
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        del game_sessions[game_id]
        return {
            "state": env.get_state(),
            "status": f"Game over! Player {env.current_player_num} wins!",
            "game_over": True
        }

    # Let AI move if the next player is AI
    current_player = env.current_player_num
    if players[current_player] == "ai":
        legal = env.legal_actions()

        # Flatten the observation for the model input
        obs_input = obs.reshape(1, *obs.shape)
        # Let the model predict the best action index
        action_idx, _ = games[game]["model"].predict(obs_input, deterministic=True)
        ai_action = tuple(action_idx[0])

        # Check if it's legal
        if ai_action not in legal:
            ai_action = legal[np.random.choice(len(legal))]

        # logger.debug(f"AI action: {ai_action}")
        obs, reward, terminated, truncated, info = env.step(ai_action)

        if terminated or truncated:
            del game_sessions[game_id]
            return {
                "state": env.get_state(),
                "status": f"Game over! AI wins!" if reward > 0 else "Game over! It's a draw!",
                "game_over": True
            }

    return {
        "state": env.get_state(),
        "status": f"Player {env.current_player_num}'s turn.",
        "game_over": False
    }

# === Static file serving ===
app.mount("/", StaticFiles(directory="static", html=True), name="static")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8001, log_level="debug")
