"""
Example script for using the CartPoleSwingUp environment.

This script demonstrates how to use the CartPoleSwingUp environment
with a random agent. The agent will not be able to solve the task,
but this shows how to interact with the environment.
"""

import time

import gymnasium as gym

import gymnasium_cartpole_swingup  # noqa: F401 - Required for environment registration


def main():
    # Create the environment with human rendering
    env = gym.make("CartPoleSwingUp-v0", render_mode="human")

    print("Environment created:")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")

    # Reset the environment and get the initial observation
    observation, info = env.reset(seed=42)
    print(f"Initial observation: {observation}")

    # Run for a fixed number of steps
    total_reward = 0

    for step in range(1000):
        # Sample a random action (replace with your agent's policy)
        action = env.action_space.sample()

        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # Print status every 100 steps
        if step % 100 == 0:
            print(
                f"Step {step}, Current reward: {reward:.3f}, Total reward: {total_reward:.3f}"
            )

        # End the episode if terminated or truncated
        if terminated or truncated:
            print(
                f"Episode ended after {step + 1} steps with total reward: {total_reward:.3f}"
            )
            break

        # Small delay to make the rendering visible
        time.sleep(0.01)

    # Close the environment
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    main()
