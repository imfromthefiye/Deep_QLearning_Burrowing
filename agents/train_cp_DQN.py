import envs  # registers ConePenEnv
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import os
import torch
from torch import nn
import torch.nn.functional as F

# DQN backbone
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.out(x)

# Simple replay buffer
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    def append(self, transition):
        self.memory.append(transition)
    def sample(self, k):
        return random.sample(self.memory, k)
    def __len__(self):
        return len(self.memory)

class ConePenDQL:
    """DDQN trainer for ConePenEnv: 4-state, 3-action inputs."""
    # hyperparameters
    learning_rate_a    = 1e-2
    discount_factor_g  = 0.9
    network_sync_rate  = 50_000
    replay_memory_size = 100_000
    mini_batch_size    = 32

    loss_fn = nn.MSELoss()
    optimizer = None

    # Environment specs
    ENV_NAME    = "ConePenEnv-v0"
    state_size  = 4  # [depth_ratio, tip_speed_norm, tip_force_norm, time_frac]
    action_size = 3  # {0: retract, 1: penetrate fast, 2: stop}
    hidden_units = 64

    def train(self, episodes, render=False):
        env = gym.make(self.ENV_NAME, render_mode='human' if render else None)
        memory = ReplayMemory(self.replay_memory_size)
        epsilon = 1.0

        # build networks
        policy_dqn = DQN(self.state_size, self.hidden_units, self.action_size)
        target_dqn = DQN(self.state_size, self.hidden_units, self.action_size)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        rewards_per_episode = []
        epsilon_history     = []
        step_count = 0
        best_reward = -np.inf

        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                # ε-greedy action selection
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(obs)).argmax().item()

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # store transition
                memory.append((obs, action, next_obs, reward, done))
                obs = next_obs
                total_reward += reward
                step_count += 1

                # sync target network periodically
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

            # record performance
            rewards_per_episode.append(total_reward)
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(policy_dqn.state_dict(), f"conepen_dql_best_ep{ep}.pt")

            # learn from experiences
            if len(memory) >= self.mini_batch_size:
                batch = memory.sample(self.mini_batch_size)
                self.optimize(batch, policy_dqn, target_dqn)
                epsilon = max(epsilon - 1/episodes, 0.0)
                epsilon_history.append(epsilon)

            # periodic logging
            if ep and ep % 100 == 0:
                print(f"Episode {ep} | ε={epsilon:.3f} | Best={best_reward:.2f}")
                self.plot_progress(rewards_per_episode, epsilon_history)

        env.close()

    def optimize(self, batch, policy_dqn, target_dqn):
        current_qs, target_qs = [], []
        for s, a, s2, r, done in batch:
            s_t = self.state_to_dqn_input(s)
            current_q = policy_dqn(s_t)
            current_qs.append(current_q)

            if done:
                target_val = torch.tensor([r])
            else:
                with torch.no_grad():
                    best_a = policy_dqn(self.state_to_dqn_input(s2)).argmax().item()
                    target_val = (r + self.discount_factor_g *
                                  target_dqn(self.state_to_dqn_input(s2))[best_a])
                    target_val = torch.tensor([target_val])

            target_q = target_dqn(s_t).clone()
            target_q[a] = target_val
            target_qs.append(target_q)

        loss = self.loss_fn(torch.stack(current_qs), torch.stack(target_qs))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_to_dqn_input(self, state) -> torch.Tensor:
        return torch.FloatTensor(state)

    def plot_progress(self, rewards, eps_hist):
        plt.figure(figsize=(8,3))
        plt.subplot(1,2,1)
        plt.plot(rewards); plt.title("Episode Reward")
        plt.subplot(1,2,2)
        plt.plot(eps_hist); plt.title("Epsilon Decay")
        plt.tight_layout()
        plt.savefig("conepen_dql_progress.png")

    def test(self, episodes, model_path):
        env = gym.make(self.ENV_NAME, render_mode="human")
        policy = DQN(self.state_size, self.hidden_units, self.action_size)
        policy.load_state_dict(torch.load(model_path))
        policy.eval()

        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    action = policy(self.state_to_dqn_input(obs)).argmax().item()
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

        env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train or test ConePen DQN agent')
    parser.add_argument("--episodes", type=int, default=10000,
                        help="number of training episodes")
    parser.add_argument("--render", action="store_true",
                        help="render environment during training")
    parser.add_argument("--test", action="store_true",
                        help="run in test mode instead of training")
    parser.add_argument("--model", type=str, default="conepen_dql_best.pt",
                        help="path to model file for testing")
    args = parser.parse_args()

    agent = ConePenDQL()

    if args.test:
        if not os.path.exists(args.model):
            print(f"Model file {args.model} not found. Train first.")
        else:
            print(f"Testing with model {args.model}")
            agent.test(5, args.model)
    else:
        print(f"Training for {args.episodes} episodes (render={args.render})")
        agent.train(args.episodes, render=args.render)
