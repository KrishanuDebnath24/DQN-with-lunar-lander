# Imports
from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import math

# Fix for NumPy 1.24+ compatibility (np.bool8 deprecation)
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# Hyperparameters for training and replay buffer
gamma = 0.99  # Discount factor
batch_size = 64
buffer_size = 50000  # Max size of replay buffer
min_replay_size = 1000  # Minimum transitions before training starts
epsilon_start = 1.0  # Initial epsilon for epsilon-greedy
epsilon_end = 0.1  # Final epsilon
epsilon_decay = 40000  # Steps over which to decay epsilon
target_update_freq = 1000  # Frequency to update target network

# Prioritized Experience Replay (PER) parameters
alpha = 0.6       # Controls how much prioritization is used
beta_start = 0.4  # Initial beta for importance sampling
beta_frames = 1000000  # Anneal beta to 1.0 over this many steps
prior_eps = 1e-6  # Small constant to avoid zero priority

# Prioritized Replay Buffer implementation
class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha):
        self.max_size = max_size
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((max_size,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, transition):
        # Set priority to max of current priorities or 1.0 if buffer is empty
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, beta):
        # Compute sampling probabilities
        prios = self.priorities if self.size == self.max_size else self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        # Update transition priorities
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio + prior_eps

# Simple feedforward Q-network
class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_features = int(np.prod(env.observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def act(self, obs):
        # Select greedy action from Q-values
        obs_array = np.array(obs, dtype=np.float32)
        obs_t = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        q_values = self(obs_t)
        return torch.argmax(q_values).item()

# Setup environment and buffers
env = gym.make('LunarLander-v2')
replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha)
reward_buffer = deque([0.0, 0.0], maxlen=100)  # Rolling average for reward

# Initialize networks and optimizer
online_network = Network(env)
target_network = Network(env)
target_network.load_state_dict(online_network.state_dict())
optimizer = torch.optim.Adam(online_network.parameters(), lr=1e-3)

# Pre-fill replay buffer with random actions
obs, _ = env.reset()
for _ in range(min_replay_size):
    action = env.action_space.sample()
    new_obs, reward, terminated, truncated, _ = env.step(action)
    done = bool(terminated or truncated)
    replay_buffer.add((obs, action, reward, done, new_obs))
    obs = new_obs
    if done:
        obs, _ = env.reset()

# Prepare data for plotting
steps_list = []
avg_reward_list = []

# Main training loop
try:
    obs, _ = env.reset()
    episode_reward = 0.0
    beta = beta_start

    for step in itertools.count():
        # Decay epsilon and beta over time
        epsilon = np.interp(step, [0, epsilon_decay], [epsilon_start, epsilon_end])
        beta = min(1.0, beta_start + step * (1.0 - beta_start) / beta_frames)

        # Choose action using epsilon-greedy policy
        if random.random() <= epsilon:
            action = env.action_space.sample()
        else:
            action = online_network.act(obs)

        # Step environment
        new_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        replay_buffer.add((obs, action, reward, done, new_obs))
        obs = new_obs
        episode_reward += reward

        if done:
            obs, _ = env.reset()
            reward_buffer.append(episode_reward)
            episode_reward = 0.0

        # Start training once replay buffer is filled
        if len(replay_buffer.buffer) >= min_replay_size:
            transitions, indices, weights = replay_buffer.sample(batch_size, beta)
            weights_t = torch.as_tensor(weights, dtype=torch.float32)
            
            # Extract batch elements
            obses = np.array([t[0] for t in transitions], dtype=np.float32)
            actions = np.array([t[1] for t in transitions])
            rews = np.array([t[2] for t in transitions], dtype=np.float32)
            dones = np.array([t[3] for t in transitions], dtype=np.bool8)
            new_obses = np.array([t[4] for t in transitions], dtype=np.float32)

            # Convert to tensors
            obses_t = torch.as_tensor(obses, dtype=torch.float32)
            actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
            rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
            dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
            new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

            # Compute Double DQN targets
            with torch.no_grad():
                sel_actions = online_network(new_obses_t).argmax(dim=1, keepdim=True)
                targ_q_vals = target_network(new_obses_t).gather(1, sel_actions)
                targets = rews_t + gamma * (1 - dones_t) * targ_q_vals

            # Compute current Q estimates and TD error
            q_vals = online_network(obses_t)
            q_pred = torch.gather(q_vals, dim=1, index=actions_t)
            td_errors = (q_pred - targets).abs().detach().squeeze().numpy()
            loss = (weights_t * nn.functional.smooth_l1_loss(q_pred, targets, reduction='none')).mean()

            # Optimize network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update priorities in buffer
            replay_buffer.update_priorities(indices, td_errors + prior_eps)

            # Update target network
            if step % target_update_freq == 0:
                target_network.load_state_dict(online_network.state_dict())

        # Periodic logging
        if step % 1000 == 0 and len(reward_buffer) > 0:
            avg = np.mean(reward_buffer)
            print(f"Step: {step}, Avg Reward (last 100 eps): {avg:.2f}")
            steps_list.append(step)
            avg_reward_list.append(avg)

except KeyboardInterrupt:
    print("\nTraining interrupted by user; saving plot...")

finally:
    # Plot training performance
    plt.figure(figsize=(10, 6))
    plt.plot(steps_list, avg_reward_list)
    max_step = max(steps_list) if steps_list else 0
    x_ticks = np.arange(0, max_step + 50000, 50000)
    plt.xticks(x_ticks)
    min_reward = min(avg_reward_list) if avg_reward_list else 0
    max_reward = max(avg_reward_list) if avg_reward_list else 0
    y_ticks = np.arange(np.floor(min_reward / 50) * 50, np.ceil(max_reward / 50) * 50 + 50, 50)
    plt.yticks(y_ticks)
    plt.xlabel('Steps')
    plt.ylabel('Avg Reward')
    plt.title('Training Progress Double DQN with PER')
    plt.grid(True)
    os.makedirs('plots', exist_ok=True)
    out_file = os.path.join('plots', 'doubledqn_per.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {os.path.abspath(out_file)}")
    plt.show()
    env.close()
