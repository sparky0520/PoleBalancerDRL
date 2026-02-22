import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Initialise the environment
env = gym.make("CartPole-v1", render_mode="human")

# Create the DQNAgent
model = DQN(
    policy="MlpPolicy", # Multi-layer Perceptron policy (standard for simple environments).
    env=env,
    learning_rate=0.001,
    buffer_size=50000,
    learning_starts=10,
    target_update_interval=100, # Approximate target_model_update=0.01
    verbose=1
)

# Train the agent
print("Starting training...")
model.learn(total_timesteps=100000, progress_bar=True)
print("Training finished.")

# Evaluate the agent
print("Evaluating the agent...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Close the environment
env.close()