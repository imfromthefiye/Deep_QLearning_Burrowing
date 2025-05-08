import envs  # registers AnchorExpEnv
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import os
import torch
from torch import nn
import torch.nn.functional as F
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordVideo
from datetime import datetime

# DQN backbone with two hidden layers [128, 64]
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, h2_nodes, out_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)
        self.out = nn.Linear(h2_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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

class AnchorExpDQL:
    """DDQN trainer for AnchorExpEnv: 4-state, 3-action inputs."""
    # hyperparameters
    learning_rate_a    = 3e-4
    discount_factor_g  = 0.9
    network_sync_rate  = 50_000
    replay_memory_size = 100_000
    mini_batch_size    = 32

    loss_fn = nn.MSELoss()
    optimizer = None

    # Environment specs
    ENV_NAME    = "AnchorExpEnv-v0"
    state_size  = 4  # [expansion_ratio, velocity_of_anchor, horizontal_reaction_force, elapsed_time_fraction]
    action_size = 3  # {0: expand (+v_norm), 1: contract (-v_norm), 2: stop (0)}
    hidden_units = [128, 64]  # Two hidden layers for better representation
    
    # Reward function parameters (r_t = αΔe_t - βΔF_t - γ_t - δ1(e_t>1) + η1(e_t≥1))
    # α: progress reward coefficient
    # β: force penalty coefficient
    # γ: time penalty
    # δ: overshoot penalty
    # η: success bonus

    def make_env(self, idx=None):
        """Factory function to create single environment instance
        Args:
            idx: If 0, creates a recording environment, otherwise standard env
        """
        try:
            if idx == 0:  # First environment will record
                # Create recording directory
                recording_path = "G:\\My Drive\\results\\recordings"
                os.makedirs(recording_path, exist_ok=True)
                
                # Create env with video recording
                base_env = gym.make(self.ENV_NAME, render_mode="rgb_array")
                wrapped_env = RecordVideo(
                    base_env,
                    recording_path,
                    episode_trigger=lambda x: x % 100 == 0,  # Record every 100th episode
                    name_prefix=f"anchorexp_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                # Test reset to ensure environment works
                wrapped_env.reset()
                return wrapped_env
            else:
                env = gym.make(self.ENV_NAME, render_mode=None)
                # Test reset to ensure environment works
                env.reset()
                return env
        except Exception as e:
            print(f"Error creating environment (idx={idx}): {str(e)}")
            raise

    def train(self, episodes, render=False):
        try:
            # Create vectorized environment with first env recording
            num_envs = 4  # Number of parallel environments
            vec_env = SyncVectorEnv([
                lambda: self.make_env(idx) for idx in range(num_envs)
            ])
            
            # Test initial reset
            obs, _ = vec_env.reset()
            print(f"Initial observation shape: {obs.shape}")
            
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = 1.0

            # build networks with two hidden layers [128, 64]
            policy_dqn = DQN(self.state_size, self.hidden_units[0], self.hidden_units[1], self.action_size)
            target_dqn = DQN(self.state_size, self.hidden_units[0], self.hidden_units[1], self.action_size)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            rewards_per_episode = []
            epsilon_history = []
            step_count = 0
            best_reward = -np.inf
            
            # Data collection for detailed analysis
            expansion_history = []
            force_history = []
            velocity_history = []
            time_history = []

            for ep in range(episodes):
                obs, _ = vec_env.reset()
                dones = np.zeros(num_envs, dtype=bool)
                total_rewards = np.zeros(num_envs)
                episode_step_count = 0
                
                # For tracking values within an episode (now per environment)
                episode_expansions = [[] for _ in range(num_envs)]
                episode_forces = [[] for _ in range(num_envs)]
                episode_velocities = [[] for _ in range(num_envs)]
                episode_times = [[] for _ in range(num_envs)]

                while not dones.all():
                    # ε-greedy action selection (vectorized)
                    if random.random() < epsilon:
                        actions = vec_env.action_space.sample()
                    else:
                        with torch.no_grad():
                            actions = policy_dqn(torch.FloatTensor(obs)).argmax(dim=1).numpy()

                    # Record metrics from observations (for each env)
                    for i in range(num_envs):
                        if not dones[i]:
                            episode_expansions[i].append(obs[i][0])
                            episode_velocities[i].append(obs[i][1])
                            episode_forces[i].append(obs[i][2])
                            episode_times[i].append(obs[i][3])
                    
                    next_obs, rewards, terminated, truncated, _ = vec_env.step(actions)
                    dones = terminated | truncated
                    episode_step_count += 1

                    # Store transitions for non-done environments
                    for i in range(num_envs):
                        if not dones[i]:
                            memory.append((obs[i], actions[i], next_obs[i], rewards[i], dones[i]))
                            total_rewards[i] += rewards[i]
                    
                    obs = next_obs
                    step_count += num_envs

                    # sync target network periodically
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

                # Record metrics for each completed environment
                for i in range(num_envs):
                    rewards_per_episode.append(total_rewards[i])
                    expansion_history.append(max(episode_expansions[i]))
                    force_history.append(max(episode_forces[i]))
                    velocity_history.append(np.mean(np.abs(episode_velocities[i])))
                    time_history.append(episode_step_count / vec_env.envs[0].MAX_STEPS)
                    
                    if total_rewards[i] > best_reward:
                        best_reward = total_rewards[i]
                        torch.save(policy_dqn.state_dict(), "G:\\My Drive\\results\\anchorexp_dql_best.pt")
                        torch.save(policy_dqn.state_dict(), f"G:\\My Drive\\results\\anchorexp_dql_best_ep{ep}.pt")

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
                    
                    # Save data for later analysis
                    np.savez("G:\\My Drive\\results\\anchorexp_training_data.npz",
                        rewards=rewards_per_episode,
                        epsilon=epsilon_history,
                        expansion=expansion_history,
                        force=force_history,
                        velocity=velocity_history,
                        time=time_history,
                        episode=range(len(rewards_per_episode)))

            # Save final data
            np.savez("G:\\My Drive\\results\\anchorexp_training_data.npz",
                rewards=rewards_per_episode,
                epsilon=epsilon_history,
                expansion=expansion_history,
                force=force_history,
                velocity=velocity_history,
                time=time_history,
                episode=range(len(rewards_per_episode)))

        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
        finally:
            if 'vec_env' in locals():
                vec_env.close()

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
        
        # Clip gradients by norm to prevent exploding TD errors
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=1.0)
        
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
        plt.savefig("G:\\My Drive\\results\\anchorexp_dql_progress.png")

    def test(self, episodes, model_path):
        env = gym.make(self.ENV_NAME, render_mode="human")
        policy = DQN(self.state_size, self.hidden_units[0], self.hidden_units[1], self.action_size)
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
    parser = argparse.ArgumentParser(description='Train or test AnchorExp DQN agent')
    parser.add_argument("--episodes", type=int, default=10000,
                        help="number of training episodes")
    parser.add_argument("--render", action="store_true",
                        help="render environment during training")
    parser.add_argument("--test", action="store_true",
                        help="run in test mode instead of training")
    parser.add_argument("--model", type=str, default="anchorexp_dql_best.pt",
                        help="path to model file for testing")
    args = parser.parse_args()

    agent = AnchorExpDQL()

    if args.test:
        if not os.path.exists(args.model):
            print(f"Model file {args.model} not found. Train first.")
        else:
            print(f"Testing with model {args.model}")
            agent.test(5, args.model)
    else:
        print(f"Training for {args.episodes} episodes (render={args.render})")
        agent.train(args.episodes, render=args.render)
