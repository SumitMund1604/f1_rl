import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from f1_game import F1Game


class F1RacingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self, 
        render_mode: Optional[str] = None,
        width: int = 800,
        height: int = 600,
        max_episode_steps: int = 2000
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.game = F1Game(width=width, height=height)
        self.game.max_steps = max_episode_steps
        
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(15,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        
        if self.render_mode == "human":
            self.game.enable_render()
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        observation = self.game.reset()
        info = {"lap_count": 0, "checkpoint": 0}
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward, done, game_info = self.game.step(action)
        observation = self.game.get_state()
        
        terminated = done and (reward == -100)
        truncated = done and (reward != -100)
        
        info = {
            "lap_count": game_info.get("score", 0),
            "checkpoint": game_info.get("checkpoints", 0),
            "steps": game_info.get("steps", 0)
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "human":
            self.game.render_with_fps(self.metadata["render_fps"])
            return None
        elif self.render_mode == "rgb_array":
            return self.game.render(mode="rgb_array")
        return None
    
    def close(self):
        self.game.disable_render()


def register_env():
    gym.register(
        id="F1Racing-v0",
        entry_point="f1_env:F1RacingEnv",
        max_episode_steps=2000
    )


if __name__ == "__main__":
    register_env()
    env = gym.make("F1Racing-v0", render_mode="human")
    obs, info = env.reset()
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
