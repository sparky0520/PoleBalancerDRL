import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Initialise the training environment (no rendering for speed)
env = gym.make("CartPole-v1")

# Create the DQNAgent with more robust hyperparameters
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,           # Lower learning rate to prevent loss explosion
    buffer_size=100000,           # Match total timesteps
    learning_starts=1000,         # Collect data before training
    target_update_interval=500,   # Slower target updates for stability
    batch_size=64,                # Larger batch size
    exploration_fraction=0.5,     # Explore for 50% of the training time
    exploration_final_eps=0.01,   # Final exploration rate
    verbose=1
)

# Train the agent
print("Starting training...")
# Verification: 100,000 steps to ensure it reaches higher rewards
model.learn(total_timesteps=100000, progress_bar=True)
print("Training finished.")

# Save the model
model.save("dqn_cartpole")
print("Model saved as dqn_cartpole.zip")

# Close training environment
env.close()

# Evaluate the agent
print("\nEvaluating the agent...")
# Create a new environment for evaluation with rendering enabled
eval_env = gym.make("CartPole-v1", render_mode="human")

# Use evaluate_policy to get individual episode rewards
rewards, lengths = evaluate_policy(
    model, 
    eval_env, 
    n_eval_episodes=10, 
    render=True, 
    return_episode_rewards=True
)

print("\nEvaluation Results:")
for i, reward in enumerate(rewards):
    print(f"Episode {i+1}: Reward = {reward}")

mean_reward = np.mean(rewards)
std_reward = np.std(rewards)
print(f"\nMean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Close evaluation environment
eval_env.close()