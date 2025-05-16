import gymnasium as gym
from stable_baselines3 import PPO

# Create environment
env = gym.make("LunarLander-v2", render_mode="human")

# Initialize PPO without TensorBoard logging
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=6e-4,
    n_steps=2048,
    batch_size=256,
    gamma=0.99,
    tensorboard_log=None  # Disable logging
)

# Train
model.learn(total_timesteps=200_000)
model.save("ppo_lunar_lander")

# Test
observation, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    env.render()

env.close()
