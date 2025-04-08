"""Tests for the CartPoleSwingUp environment."""

import gymnasium as gym
import numpy as np
import pytest

import gymnasium_cartpole_swingup  # noqa: F401 - Required for environment registration


def test_environment_creation():
    """Test that the environment can be created."""
    env = gym.make("CartPoleSwingUp-v0")
    assert env is not None


def test_reset():
    """Test the reset method."""
    env = gym.make("CartPoleSwingUp-v0")
    observation, info = env.reset(seed=42)

    # Check observation shape
    assert observation.shape == (5,)
    assert isinstance(info, dict)

    # Check that pole starts in downward position (θ ≈ π)
    # sine of π should be 0, cosine of π should be -1
    cos_theta = observation[2]
    sin_theta = observation[3]
    assert -1.1 < cos_theta < -0.9  # cos(π) ≈ -1
    assert -0.6 < sin_theta < 0.6  # sin(π) ≈ 0, with wide tolerance for randomness


def test_step():
    """Test the step method."""
    env = gym.make("CartPoleSwingUp-v0")
    env.reset(seed=42)

    action = np.array([0.5])  # Apply force to the right
    observation, reward, terminated, truncated, info = env.step(action)

    # Check return types
    assert observation.shape == (5,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    # Test termination condition
    env.reset()

    # Apply large force for many steps to push cart to boundary
    for _ in range(100):
        observation, reward, terminated, truncated, info = env.step(np.array([1.0]))
        if terminated:
            break

    # Cart should eventually go out of bounds
    assert terminated or abs(observation[0]) > 2.0


def test_render_modes():
    """Test that render modes are correct."""
    env = gym.make("CartPoleSwingUp-v0")
    assert "human" in env.metadata["render_modes"]
    assert "rgb_array" in env.metadata["render_modes"]


def test_render_rgb_array():
    """Test that rgb_array rendering works."""
    env = gym.make("CartPoleSwingUp-v0", render_mode="rgb_array")
    env.reset()

    # Get a frame
    img = env.render()

    # Should be a numpy array with shape (height, width, 3)
    assert isinstance(img, np.ndarray)
    assert img.shape[2] == 3  # RGB channels
    assert img.shape[0] > 0  # Height
    assert img.shape[1] > 0  # Width
