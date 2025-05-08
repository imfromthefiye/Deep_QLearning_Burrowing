import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from collections import deque

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordVideo

# Register your AnchorExpEnv by importing your envs package
import envs  # <— must register AnchorExpEnv-v0 via entry_point in setup.py or envs/__init__.py

# ===============================================
# DQN network
# ===============================================
class DQN(nn.Module):
    def __init__(self, in_dim, h1, h2, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# ===============================================
# Simple replay buffer
# ===============================================
class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ===============================================
# Anchor Expansion Trainer
# ===============================================
class AnchorExpTrainer:
    ENV_ID = "AnchorExpEnv-v0"
    STATE_DIM = 4    # [expansion_ratio, speed, force, time_frac]
    ACTION_DIM = 3   # {0: expand, 1: contract, 2: stop}

    # Hyperparameters
    LR = 3e-4
    GAMMA = 0.9
    SYNC_EVERY = 50_000
    MEM_CAP = 100_000
    BATCH = 32
    MAX_STEPS_EP = 500

    def __init__(self, results_root="G:/My Drive/results"):
        # ensure results dirs exist
        self.results_root = results_root
        self.video_dir = os.path.join(results_root, "recordings")
        os.makedirs(self.video_dir, exist_ok=True)

        # build networks
        self.policy = DQN(self.STATE_DIM, 128, 64, self.ACTION_DIM)
        self.target = DQN(self.STATE_DIM, 128, 64, self.ACTION_DIM)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.LR)

        # replay memory
        self.memory = ReplayMemory(self.MEM_CAP)
        self.steps_since_sync = 0

        # trackers
        self.rewards = []
        self.epsilons = []
        self.expansions = []
        self.forces = []
        self.vels = []
        self.times = []

    def make_vec_env(self, n_envs):
        """Create a SyncVectorEnv and wrap the first env for video recording."""
        def make_one(_):
            return gym.make(self.ENV_ID, render_mode=None)

        vec = SyncVectorEnv([lambda idx=i: make_one(i) for i in range(n_envs)])

        # wrap only the first environment in RecordVideo
        def make_recorded(_):
            base = gym.make(self.ENV_ID, render_mode="rgb_array")
            return RecordVideo(
                base,
                video_folder=self.video_dir,
                name_prefix="anchor_exp"
            )

        # replace the first env-fn
        vec.env_fns[0] = lambda: make_recorded(0)
        return vec

    def select_action(self, obs, eps):
        if random.random() < eps:
            return np.random.randint(self.ACTION_DIM, size=obs.shape[0])
        with torch.no_grad():
            q = self.policy(torch.FloatTensor(obs))
            return q.argmax(dim=-1).cpu().numpy()

    def compute_loss(self, batch):
        states, actions, next_states, rewards, dones = zip(*batch)
        s  = torch.FloatTensor(states)
        ns = torch.FloatTensor(next_states)
        a  = torch.LongTensor(actions).unsqueeze(-1)
        r  = torch.FloatTensor(rewards).unsqueeze(-1)
        d  = torch.FloatTensor(dones).unsqueeze(-1)

        qvals = self.policy(s).gather(1, a)
        with torch.no_grad():
            next_q = self.target(ns).max(1, keepdim=True)[0]
            tgt_q = r + (1 - d) * self.GAMMA * next_q

        return nn.MSELoss()(qvals, tgt_q)

    def train(self, episodes=2000, n_envs=4):
        envs = self.make_vec_env(n_envs)
        eps = 1.0

        for ep in range(1, episodes + 1):
            obs, _ = envs.reset()
            ep_rewards = np.zeros(n_envs)
            exp_hist, force_hist, vel_hist, time_hist = ([] for _ in range(4))

            for step in range(self.MAX_STEPS_EP):
                acts = self.select_action(obs, eps)
                nxt, rews, terms, truncs, _ = envs.step(acts)

                # store transitions & metrics
                for i in range(n_envs):
                    self.memory.append(
                        (obs[i], acts[i], nxt[i], rews[i], float(terms[i] or truncs[i]))
                    )
                    ep_rewards[i] += rews[i]

                    exp_hist.append(nxt[i][0])
                    vel_hist.append(abs(nxt[i][1]))
                    force_hist.append(nxt[i][2])
                    time_hist.append(nxt[i][3])

                obs = nxt

                # learn
                if len(self.memory) >= self.BATCH:
                    batch = self.memory.sample(self.BATCH)
                    loss = self.compute_loss(batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    self.optimizer.step()

                    self.steps_since_sync += 1
                    if self.steps_since_sync >= self.SYNC_EVERY:
                        self.target.load_state_dict(self.policy.state_dict())
                        self.steps_since_sync = 0

                if all(terms) or all(truncs):
                    break

            # record end‐of‐episode data
            mean_r = ep_rewards.mean()
            self.rewards.append(mean_r)
            self.epsilons.append(eps)
            self.expansions.append(np.max(exp_hist))
            self.forces.append(np.max(force_hist))
            self.vels.append(np.mean(vel_hist))
            self.times.append(len(time_hist) / (step + 1))

            # epsilon decay
            eps = max(0.0, eps - 1.0 / episodes)

            if ep % 100 == 0:
                print(f"Ep {ep} | ε={eps:.3f} | R={mean_r:.2f}")

            # periodic checkpoint
            if ep % 500 == 0:
                torch.save(
                    self.policy.state_dict(),
                    os.path.join(self.results_root, f"anchor_policy_ep{ep}.pt")
                )

        envs.close()

        # final save
        np.savez(
            os.path.join(self.results_root, "anchor_metrics.npz"),
            rewards=self.rewards,
            epsilons=self.epsilons,
            expansions=self.expansions,
            forces=self.forces,
            velocities=self.vels,
            times=self.times,
        )
        print("Done. Metrics + videos written to:", self.results_root)

# ===============================================
# Command line entry
# ===============================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--results", type=str, default="G:/My Drive/results")
    args = parser.parse_args()

    trainer = AnchorExpTrainer(results_root=args.results)
    trainer.train(episodes=args.episodes, n_envs=args.envs)
