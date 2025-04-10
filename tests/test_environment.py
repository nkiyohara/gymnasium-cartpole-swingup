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
    assert observation.shape == (4,)
    assert isinstance(info, dict)

    # Check that pole starts in downward position (θ ≈ π)
    x, x_dot, theta, theta_dot = observation
    assert -np.pi - 0.5 < theta < -np.pi + 0.5 or np.pi - 0.5 < theta < np.pi + 0.5  # theta ≈ π


def test_step():
    """Test the step method."""
    env = gym.make("CartPoleSwingUp-v0")
    env.reset(seed=42)

    action = np.array([0.5])  # Apply force to the right
    observation, reward, terminated, truncated, info = env.step(action)

    # Check return types
    assert observation.shape == (4,)
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


def test_cost_modes():
    """Test that both cost modes (default and pilco) work correctly."""
    # Test default cost mode
    env_default = gym.make("CartPoleSwingUp-v0", cost_mode="default")
    env_default.reset(seed=42)
    action = np.array([0.0])  # No force
    obs_default, reward_default, _, _, _ = env_default.step(action)
    
    # Test pilco cost mode
    env_pilco = gym.make("CartPoleSwingUp-v0", cost_mode="pilco", sigma_c=0.25)
    env_pilco.reset(seed=42)  # Same seed as default
    obs_pilco, reward_pilco, _, _, _ = env_pilco.step(action)
    
    # Observations should be the same with same seed and action
    np.testing.assert_allclose(obs_default, obs_pilco)
    
    # Rewards should be different between modes
    assert reward_default != reward_pilco
    
    # Test invalid cost mode
    with pytest.raises(ValueError):
        env_invalid = gym.make("CartPoleSwingUp-v0", cost_mode="invalid")
        env_invalid.reset()
        env_invalid.step(np.array([0.0]))


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
