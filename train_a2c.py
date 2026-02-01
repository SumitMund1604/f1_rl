import argparse
import os
import sys
from typing import Tuple

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f1_env import F1RacingEnv

gym.register(id="F1Racing-v0", entry_point="f1_env:F1RacingEnv", max_episode_steps=2000)


class A2C(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        device: torch.device,
        critic_lr: float = 0.005,
        actor_lr: float = 0.001,
        n_envs: int = 1
    ):
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        
        self.critic = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)
        
        self.actor = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        ).to(device)
        
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
    
    def forward(self, x: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.FloatTensor(x).to(self.device)
        state_values = self.critic(x)
        action_logits = self.actor(x)
        return state_values, action_logits
    
    def select_action(self, x: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_values, action_logits = self.forward(x)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        actions = action_dist.sample()
        action_log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return actions, action_log_probs, state_values, entropy
    
    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lam: float,
        ent_coef: float,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=device)
        
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae
        
        critic_loss = advantages.pow(2).mean()
        actor_loss = -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        
        return critic_loss, actor_loss
    
    def update_parameters(self, critic_loss: torch.Tensor, actor_loss: torch.Tensor):
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()


def plot_training(scores, mean_scores, save_path=None):
    plt.clf()
    plt.title('F1 Racing RL - Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores, label='Score', alpha=0.6)
    plt.plot(mean_scores, label='Mean Score (last 50)', linewidth=2)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path)
    plt.pause(0.001)


def train(args):
    print("=" * 60)
    print("F1 Racing RL - A2C Training")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    print(f"Creating {args.n_envs} parallel environments...")
    envs = gym.make_vec("F1Racing-v0", num_envs=args.n_envs, max_episode_steps=args.max_steps)
    
    obs_shape = envs.single_observation_space.shape[0]
    action_shape = envs.single_action_space.n
    print(f"Observation space: {obs_shape}")
    print(f"Action space: {action_shape}")
    
    agent = A2C(
        n_features=obs_shape,
        n_actions=action_shape,
        device=device,
        critic_lr=args.critic_lr,
        actor_lr=args.actor_lr,
        n_envs=args.n_envs
    )
    
    envs_wrapper = gym.wrappers.vector.RecordEpisodeStatistics(envs, buffer_length=args.n_envs * args.updates)
    
    critic_losses, actor_losses, entropies = [], [], []
    episode_scores, mean_scores = [], []
    
    gamma = args.gamma
    lam = args.lam
    ent_coef = args.ent_coef
    n_steps = args.n_steps
    
    if args.plot:
        plt.ion()
        plt.figure(figsize=(10, 6))
    
    print(f"\nTraining for {args.updates} updates...")
    print(f"Steps per update: {n_steps}, Envs: {args.n_envs}")
    print("-" * 60)
    
    states, info = envs_wrapper.reset(seed=42)
    
    for update in tqdm(range(args.updates)):
        ep_value_preds = torch.zeros(n_steps, args.n_envs, device=device)
        ep_rewards = torch.zeros(n_steps, args.n_envs, device=device)
        ep_action_log_probs = torch.zeros(n_steps, args.n_envs, device=device)
        ep_entropy = torch.zeros(n_steps, args.n_envs, device=device)
        masks = torch.zeros(n_steps, args.n_envs, device=device)
        
        for step in range(n_steps):
            actions, action_log_probs, state_values, entropy = agent.select_action(states)
            states, rewards, terminated, truncated, infos = envs_wrapper.step(actions.cpu().numpy())
            
            ep_value_preds[step] = state_values.squeeze()
            ep_rewards[step] = torch.tensor(rewards, device=device)
            ep_action_log_probs[step] = action_log_probs
            ep_entropy[step] = entropy
            masks[step] = torch.tensor([not t for t in terminated], device=device, dtype=torch.float32)
        
        critic_loss, actor_loss = agent.get_losses(
            ep_rewards, ep_action_log_probs, ep_value_preds, ep_entropy,
            masks, gamma, lam, ent_coef, device
        )
        agent.update_parameters(critic_loss, actor_loss)
        
        critic_losses.append(critic_loss.detach().cpu().numpy())
        actor_losses.append(actor_loss.detach().cpu().numpy())
        entropies.append(ep_entropy.mean().detach().cpu().numpy())
        
        if len(envs_wrapper.return_queue) > 0:
            recent_returns = list(envs_wrapper.return_queue)[-args.n_envs:]
            avg_return = np.mean(recent_returns) if recent_returns else 0
            episode_scores.append(avg_return)
            mean_score = np.mean(episode_scores[-50:]) if episode_scores else 0
            mean_scores.append(mean_score)
        
        if update % 20 == 0 or update == args.updates - 1:
            avg_return = episode_scores[-1] if episode_scores else 0
            mean_return = mean_scores[-1] if mean_scores else 0
            print(f"Update {update:4d} | Return: {avg_return:7.1f} | Mean: {mean_return:7.1f} | "
                  f"C_Loss: {critic_loss:.3f} | A_Loss: {actor_loss:.3f}")
        
        if args.plot and update % 10 == 0 and episode_scores:
            plot_path = os.path.join(args.save_dir, 'training_progress.png')
            plot_training(episode_scores, mean_scores, save_path=plot_path)
        
        if update % 100 == 0 and update > 0:
            save_path = os.path.join(args.save_dir, 'f1_a2c_checkpoint.pth')
            torch.save({'actor': agent.actor.state_dict(), 'critic': agent.critic.state_dict(), 'update': update}, save_path)
    
    print("-" * 60)
    print("Training complete!")
    if mean_scores:
        print(f"Final mean return: {mean_scores[-1]:.2f}")
    
    final_path = os.path.join(args.save_dir, 'f1_a2c_final.pth')
    torch.save({'actor': agent.actor.state_dict(), 'critic': agent.critic.state_dict(), 'updates': args.updates}, final_path)
    print(f"Model saved to {final_path}")
    
    if args.plot and episode_scores:
        plt.ioff()
        plot_path = os.path.join(args.save_dir, 'training_final.png')
        plot_training(episode_scores, mean_scores, save_path=plot_path)
        plt.show()
    
    envs.close()


def main():
    parser = argparse.ArgumentParser(description='Train F1 Racing RL Agent with A2C')
    
    parser.add_argument('--updates', type=int, default=500)
    parser.add_argument('--n_envs', type=int, default=8)
    parser.add_argument('--n_steps', type=int, default=128)
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--ent_coef', type=float, default=0.01)
    parser.add_argument('--actor_lr', type=float, default=0.0003)
    parser.add_argument('--critic_lr', type=float, default=0.001)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--no-plot', action='store_false', dest='plot')
    parser.add_argument('--save_dir', type=str, default='models')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
