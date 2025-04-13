"""
Gymnasium CartPole SwingUp - A challenging variant of the classic CartPole environment.

This package provides a more difficult version of the CartPole environment
where the pole starts in a downward position and must be swung up and balanced.
"""

from gymnasium.envs.registration import register

from gymnasium_cartpole_swingup.cartpole_swingup import CartPoleSwingUpEnv

# Register the environment with Gymnasium
register(
    id="CartPoleSwingUp-v0",
    entry_point="gymnasium_cartpole_swingup.cartpole_swingup:CartPoleSwingUpEnv",
    max_episode_steps=1000,
)

# Explicitly export variables and classes to help with linting and import detection
__all__ = ["CartPoleSwingUpEnv"]

# Version is defined here as the single source of truth
# When updating version, only change it here
__version__ = "0.1.7"
