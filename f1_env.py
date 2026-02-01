"""
F1 Racing Environment - Gymnasium-compatible environment for RL training

This environment provides a custom F1 racing simulation where an agent
learns to navigate an oval track, avoid obstacles, and pass checkpoints.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from f1_game import F1Game


class F1RacingEnv(gym.Env):
    """
    F1 Racing Environment following Gymnasium API.
    
    Observation Space (15 dimensions):
        [0-4]:   5 wall distance sensors (normalized 0-1)
        [5-6]:   Car velocity (vx, vy) normalized
        [7]:     Car angle normalized (-1 to 1)
        [8-9]:   Direction to next checkpoint
        [10-14]: 5 obstacle proximity sensors
    
    Action Space (5 discrete actions):
        0: Accelerate forward
        1: Turn left + accelerate
        2: Turn right + accelerate
        3: Brake
        4: Coast (no action)
    
    Rewards:
        +50:   Pass checkpoint
        +200:  Complete lap
        -100:  Crash (wall or obstacle)
        +1:    Moving toward checkpoint
        -0.1:  Time penalty per step
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60
    }
    
    def __init__(
        self, 
        render_mode: Optional[str] = None,
        width: int = 800,
        height: int = 600,
        max_episode_steps: int = 2000
    ):
        """
        Initialize F1 Racing Environment.
        
        Args:
            render_mode: "human" for window display, "rgb_array" for pixel output
            width: Game window width
            height: Game window height
            max_episode_steps: Maximum steps per episode
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.game = F1Game(width=width, height=height)
        self.game.max_steps = max_episode_steps
        
        # Define observation space (15-dimensional continuous)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32
        )
        
        # Define action space (5 discrete actions)
        self.action_space = spaces.Discrete(5)
        
        # Enable rendering if needed
        if self.render_mode == "human":
            self.game.enable_render()
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (unused)
        
        Returns:
            observation: Initial state observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        observation = self.game.reset()
        info = {
            "lap_count": 0,
            "checkpoint": 0
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-4)
        
        Returns:
            observation: New state observation
            reward: Reward from the action
            terminated: Whether episode ended (crash or lap complete)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        reward, done, game_info = self.game.step(action)
        observation = self.game.get_state()
        
        # Determine termination vs truncation
        terminated = done and (reward == -100)  # Crashed
        truncated = done and (reward != -100)   # Max steps or other
        
        info = {
            "lap_count": game_info.get("score", 0),
            "checkpoint": game_info.get("checkpoints", 0),
            "steps": game_info.get("steps", 0)
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the current frame.
        
        Returns:
            rgb_array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "human":
            self.game.render_with_fps(self.metadata["render_fps"])
            return None
        elif self.render_mode == "rgb_array":
            return self.game.render(mode="rgb_array")
        return None
    
    def close(self):
        """Clean up resources"""
        self.game.disable_render()


# Register the environment with Gymnasium
def register_env():
    """Register F1RacingEnv with Gymnasium"""
    gym.register(
        id="F1Racing-v0",
        entry_point="f1_env:F1RacingEnv",
        max_episode_steps=2000
    )


if __name__ == "__main__":
    # Test the environment
    register_env()
    
    print("Creating F1Racing environment...")
    env = gym.make("F1Racing-v0", render_mode="human")
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Run a few random steps
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("Test complete!")
