import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import pygame

import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f1_env import F1RacingEnv

gym.register(id="F1Racing-v0", entry_point="f1_env:F1RacingEnv", max_episode_steps=2000)


class Actor(nn.Module):
    def __init__(self, n_features: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def select_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            logits = self.forward(state_t)
            return logits.argmax().item()


def load_model(model_path: str, n_features: int, n_actions: int) -> Actor:
    actor = Actor(n_features, n_actions)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        actor.net.load_state_dict(checkpoint['actor'])
        print(f"Loaded model from {model_path}")
    else:
        print(f"No model found at {model_path}, using random policy")
    
    actor.eval()
    return actor


def play_manual(env):
    print("\n=== MANUAL CONTROL ===")
    print("W/Up: Accelerate | A/Left: Turn Left | D/Right: Turn Right")
    print("S/Down: Brake | Space: Coast | R: Reset | Q/ESC: Quit")
    print("=" * 30)
    
    obs, info = env.reset()
    running = True
    total_reward = 0
    
    while running:
        env.render()
        
        action = 4
        keys = pygame.key.get_pressed()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
        
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action = 0
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action = 1
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action = 2
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action = 3
        elif keys[pygame.K_SPACE]:
            action = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode ended! Reward: {total_reward:.1f}, Laps: {info['lap_count']}, Checkpoints: {info['checkpoint']}")
            obs, info = env.reset()
            total_reward = 0
    
    env.close()


def play_agent(env, actor: Actor, n_episodes: int = 5, fps: int = 60):
    print(f"\n=== AGENT DEMO ({n_episodes} episodes) ===")
    
    for episode in range(1, n_episodes + 1):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode}...")
        
        while True:
            env.render()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close()
                    return
            
            action = actor.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                print(f"Episode {episode}: Steps={steps}, Reward={total_reward:.1f}, Laps={info['lap_count']}, Checkpoints={info['checkpoint']}")
                break
    
    print("\nDemo complete!")
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Play F1 Racing')
    parser.add_argument('--model', type=str, default='final', choices=['final', 'checkpoint', 'best'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--manual', action='store_true')
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--save_dir', type=str, default='models')
    
    args = parser.parse_args()
    
    env = gym.make("F1Racing-v0", render_mode="human")
    
    if args.manual:
        play_manual(env)
    else:
        if args.model_path:
            model_path = args.model_path
        elif args.model == 'final':
            model_path = os.path.join(args.save_dir, 'f1_a2c_final.pth')
        elif args.model == 'checkpoint':
            model_path = os.path.join(args.save_dir, 'f1_a2c_checkpoint.pth')
        else:
            model_path = os.path.join(args.save_dir, 'f1_a2c_best.pth')
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        actor = load_model(model_path, obs_dim, act_dim)
        
        play_agent(env, actor, args.episodes, args.fps)


if __name__ == '__main__':
    main()
