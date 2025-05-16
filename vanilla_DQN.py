# --- Imports ---
from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# --- Compatibility Fix ---
# Fix for np.bool8 deprecation in newer NumPy versions
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# --- Hyperparameters ---
gamma               = 0.99     # Discount factor for future rewards
batch_size          = 32       # Batch size for training the network
buffer_size         = 50000    # Maximum number of transitions stored in replay buffer
min_replay_size     = 1000     # Minimum number of experiences before training starts
epsilon_start       = 1.0      # Initial exploration rate
epsilon_end         = 0.02     # Final exploration rate
epsilon_decay       = 10000    # Number of steps over which epsilon decays
target_update_freq  = 1000     # Frequency (in steps) to update the target network

# --- Neural Network for Q-learning ---
class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_features = int(np.prod(env.observation_space.shape))  # Flatten input
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),                     # Hidden activation
            nn.Linear(64, env.action_space.n)  # Output layer with Q-values for each action
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        """Selects the best action given an observation (greedy policy)."""
        obs_array = np.array(obs, dtype=np.float32)
        obs_t = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        q_values = self(obs_t)
        return torch.argmax(q_values).item()  # Return action with highest Q-value

# --- Environment Setup ---
env = gym.make('LunarLander-v2')                       # Initialize environment
replay_buffer = deque(maxlen=buffer_size)              # Experience replay buffer
reward_buffer = deque([0.0, 0.0], maxlen=100)           # Track recent rewards

# --- Initialize Networks and Optimizer ---
online_network = Network(env)                          # Q-network to be trained
target_network = Network(env)                          # Target Q-network (fixed parameters)
target_network.load_state_dict(online_network.state_dict())  # Sync target with online
optimizer = torch.optim.Adam(online_network.parameters(), lr=5e-4)

# --- Pre-fill Replay Buffer with Random Actions ---
obs, _ = env.reset()
for _ in range(min_replay_size):
    action = env.action_space.sample()                 # Take random action
    new_obs, reward, terminated, truncated, _ = env.step(action)
    done = bool(terminated or truncated)
    replay_buffer.append((obs, action, reward, done, new_obs))
    obs = new_obs
    if done:
        obs, _ = env.reset()

# --- Plotting Setup ---
fig, ax = plt.subplots()
steps_list      = []  # List to track step numbers for plotting
avg_reward_list = []  # List to track average rewards

# --- Main Training Loop ---
try:
    obs, _ = env.reset()
    episode_reward = 0.0

    for step in itertools.count():
        # Linearly decay epsilon over time
        epsilon = np.interp(step, [0, epsilon_decay], [epsilon_start, epsilon_end])

        # --- Action Selection (ε-greedy) ---
        if random.random() <= epsilon:
            action = env.action_space.sample()         # Explore
        else:
            action = online_network.act(obs)           # Exploit

        # --- Step the Environment ---
        new_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        replay_buffer.append((obs, action, reward, done, new_obs))  # Store transition
        obs = new_obs
        episode_reward += reward

        if done:
            obs, _ = env.reset()
            reward_buffer.append(episode_reward)       # Track episode reward
            episode_reward = 0.0

        # --- Training Step ---
        if len(replay_buffer) >= min_replay_size:
            # Sample a mini-batch of experiences
            transitions = random.sample(replay_buffer, batch_size)
            obses     = np.array([t[0] for t in transitions], dtype=np.float32)
            actions   = np.array([t[1] for t in transitions])
            rews      = np.array([t[2] for t in transitions], dtype=np.float32)
            dones     = np.array([t[3] for t in transitions], dtype=np.bool8)
            new_obses = np.array([t[4] for t in transitions], dtype=np.float32)

            # Convert data to PyTorch tensors
            obses_t     = torch.as_tensor(obses)
            actions_t   = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
            rews_t      = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
            dones_t     = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
            new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

            # --- Compute Targets using Target Network (Vanilla DQN) ---
            with torch.no_grad():
                next_q_values = target_network(new_obses_t)
                max_next_q = next_q_values.max(1)[0].unsqueeze(-1)  # Max Q for next state
                targets = rews_t + gamma * (1 - dones_t) * max_next_q

            # --- Get Q-values for chosen actions ---
            q_vals = online_network(obses_t)
            q_pred = torch.gather(q_vals, dim=1, index=actions_t)

            # --- Compute Loss (Huber Loss) ---
            loss = nn.functional.smooth_l1_loss(q_pred, targets)

            # --- Gradient Descent Step ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Periodic Target Network Update ---
            if step % target_update_freq == 0:
                target_network.load_state_dict(online_network.state_dict())

        # --- Logging ---
        if step % 1000 == 0 and len(reward_buffer) > 0:
            avg = np.mean(reward_buffer)
            print(f"Step: {step}, Avg Reward (last 100 eps): {avg:.2f}")
            steps_list.append(step)
            avg_reward_list.append(avg)

# --- Keyboard Interrupt Handling ---
except KeyboardInterrupt:
    print("\nTraining interrupted by user; saving plot...")

# --- Final Plotting ---
finally:
    plt.figure(figsize=(10, 6))
    plt.plot(steps_list, avg_reward_list)

    # Set x-axis ticks every 50,000 steps
    max_step = max(steps_list) if steps_list else 0
    x_ticks = np.arange(0, max_step + 50000, 50000)
    plt.xticks(x_ticks)

    # Set y-axis ticks every 50 reward points
    min_reward = min(avg_reward_list) if avg_reward_list else 0
    max_reward = max(avg_reward_list) if avg_reward_list else 0
    y_ticks = np.arange(
        np.floor(min_reward / 50) * 50,
        np.ceil(max_reward / 50) * 50 + 50,
        50
    )
    plt.yticks(y_ticks)

    plt.xlabel('Steps')
    plt.ylabel('Avg Reward')
    plt.title('Training Progress DQN')  # Vanilla DQN (not Double DQN)
    plt.grid(True)

    # Save and show plot
    os.makedirs('plots', exist_ok=True)
    out_file = os.path.join('plots', 'dqn.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {os.path.abspath(out_file)}")
    plt.show()

    env.close()
