import gymnasium as gym
import numpy as np
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium import RewardWrapper

# Custom Wrapper to punish the agent for moving away from the center
class CenterRewardWrapper(RewardWrapper):
    def reward(self, reward):
        # state[0] is the cart position. Limit is ±2.4
        cart_pos = self.env.unwrapped.state[0]
        # Subtract a penalty based on distance from center (0 to 1)
        penalty = abs(cart_pos) / 2.4
        return float(reward - (penalty * 0.5)) # Ensure it returns a float

def train():
    # Initialise the training environment with the custom wrapper
    env = gym.make("CartPole-v1")
    env = CenterRewardWrapper(env)

    # Create the DQNAgent with aggressive tuning + TensorBoard logging
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        target_update_interval=500,
        train_freq=1,
        gradient_steps=1,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        tensorboard_log="./dqn_logs/",
        verbose=1
    )

    print("Starting training...")
    model.learn(total_timesteps=100000, progress_bar=True)
    print("Training finished.")

    # Save the model
    model.save("dqn_cartpole")
    print("Model saved as dqn_cartpole.zip")
    
    env.close()
    return model

def test(model_path="dqn_cartpole"):
    print(f"\nTesting the agent from {model_path}...")
    # Load the model
    model = DQN.load(model_path)
    
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

    eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test DQN on CartPole-v1")
    parser.add_argument("--test", action="store_true", help="Test the saved model instead of training")
    args = parser.parse_args()

    if args.test:
        test()
    else:
        model = train()
        # Optionally test immediately after training
        test()