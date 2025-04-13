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


def test_custom_initial_state():
    """Test setting custom initial state through reset options."""
    env = gym.make("CartPoleSwingUp-v0")
    
    # Set a specific custom initial state: cart at position 1.0, pole at 45 degrees
    custom_state = [1.0, 0.0, np.pi/4, 0.0]
    observation, info = env.reset(options={"initial_state": custom_state})
    
    # Verify that the returned observation matches our custom state
    np.testing.assert_allclose(observation, custom_state, rtol=1e-5)
    
    # Try a different initial state
    custom_state_2 = [-0.5, 0.1, np.pi/2, -0.1]
    observation, info = env.reset(options={"initial_state": custom_state_2})
    
    # Verify that the returned observation matches the second custom state
    np.testing.assert_allclose(observation, custom_state_2, rtol=1e-5)
    
    # Test with trigonometric observation mode
    env_trig = gym.make("CartPoleSwingUp-v0", obs_mode="trig")
    custom_state_3 = [0.0, 0.0, np.pi/3, 0.0]  # Cart centered, pole at 60 degrees
    observation, info = env_trig.reset(options={"initial_state": custom_state_3})
    
    # For trigonometric mode, verify that the observation correctly transforms theta
    x, x_dot, sin_theta, cos_theta, theta_dot = observation
    expected_sin = np.sin(custom_state_3[2])
    expected_cos = np.cos(custom_state_3[2])
    
    np.testing.assert_allclose(x, custom_state_3[0], rtol=1e-5)
    np.testing.assert_allclose(x_dot, custom_state_3[1], rtol=1e-5)
    np.testing.assert_allclose(sin_theta, expected_sin, rtol=1e-5)
    np.testing.assert_allclose(cos_theta, expected_cos, rtol=1e-5) 
    np.testing.assert_allclose(theta_dot, custom_state_3[3], rtol=1e-5)


def test_customized_initialization_parameters():
    """Test that custom initialization parameters work."""
    # Create environment with custom initial state distribution
    custom_mean = np.array([0.5, 0.0, 0.0, 0.0])  # Cart offset, pole up
    custom_noise = np.array([0.01, 0.01, 0.01, 0.01])  # Small noise
    
    env = gym.make(
        "CartPoleSwingUp-v0", 
        initial_state_mean=custom_mean,
        initial_state_noise=custom_noise
    )
    
    # Perform multiple resets to verify the distribution
    num_samples = 50
    states = []
    
    for _ in range(num_samples):
        observation, _ = env.reset(seed=None)  # Use different seeds
        states.append(observation)
    
    # Convert to numpy array for analysis
    states = np.array(states)
    
    # Calculate mean across all samples
    empirical_mean = np.mean(states, axis=0)
    
    # The empirical mean should be close to our custom mean (allowing some variance)
    # We use a relatively loose tolerance due to random sampling
    np.testing.assert_allclose(empirical_mean, custom_mean, rtol=0.2, atol=0.2)
    
    # Verify that the standard deviation is in the right order of magnitude
    empirical_std = np.std(states, axis=0)
    # Standard deviation might vary, but should be in similar order of magnitude
    assert all(empirical_std < custom_noise * 3), "Standard deviation too large"
    assert all(empirical_std > custom_noise * 0.1), "Standard deviation too small"


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


def test_obs_modes():
    """Test that both observation modes (raw and trig) work correctly."""
    # Test raw observation mode
    env_raw = gym.make("CartPoleSwingUp-v0", obs_mode="raw")
    obs_raw, _ = env_raw.reset(seed=42)
    
    # Raw observation should be 4-dimensional: [x, x_dot, theta, theta_dot]
    assert obs_raw.shape == (4,)
    
    # Test trigonometric observation mode
    env_trig = gym.make("CartPoleSwingUp-v0", obs_mode="trig")
    obs_trig, _ = env_trig.reset(seed=42)  # Same seed as raw
    
    # Trig observation should be 5-dimensional: [x, x_dot, sin(theta), cos(theta), theta_dot]
    assert obs_trig.shape == (5,)
    
    # Verify observation space dimensions
    assert env_raw.observation_space.shape == (4,)
    assert env_trig.observation_space.shape == (5,)
    
    # Verify the values are consistent between raw and trig modes
    x_raw, x_dot_raw, theta_raw, theta_dot_raw = obs_raw
    x_trig, x_dot_trig, sin_theta, cos_theta, theta_dot_trig = obs_trig
    
    # Position and velocities should be the same
    assert x_raw == x_trig
    assert x_dot_raw == x_dot_trig
    assert theta_dot_raw == theta_dot_trig
    
    # Verify that sin(theta) and cos(theta) match the raw theta value
    np.testing.assert_allclose(sin_theta, np.sin(theta_raw), rtol=1e-5)
    np.testing.assert_allclose(cos_theta, np.cos(theta_raw), rtol=1e-5)
    
    # Test step function with both modes
    action = np.array([0.5])
    next_obs_raw, _, _, _, _ = env_raw.step(action)
    next_obs_trig, _, _, _, _ = env_trig.step(action)
    
    # Verify shapes after step
    assert next_obs_raw.shape == (4,)
    assert next_obs_trig.shape == (5,)
    
    # Test invalid observation mode
    with pytest.raises(ValueError):
        env_invalid = gym.make("CartPoleSwingUp-v0", obs_mode="invalid")
        env_invalid.reset()


def test_custom_reward_function():
    """Test that a custom reward function can be used."""
    # Define a simple custom reward function
    def custom_reward(state, action, next_state):
        x, x_dot, theta, theta_dot = next_state
        # Simple reward based on pole angle
        return np.cos(theta) * 2.0

    # Create environment with custom reward function
    env = gym.make("CartPoleSwingUp-v0", custom_reward_fn=custom_reward)
    env.reset(seed=42)
    action = np.array([0.5])
    
    # Get reward with custom function
    _, custom_reward_value, _, _, _ = env.step(action)
    
    # Create environment with default reward function
    env_default = gym.make("CartPoleSwingUp-v0")
    env_default.reset(seed=42)
    
    # Get reward with default function
    _, default_reward_value, _, _, _ = env_default.step(action)
    
    # Rewards should be different
    assert custom_reward_value != default_reward_value
    
    # The custom reward should be exactly 2 times the cosine of theta
    # from the resulting state of the environment step
    next_state = env.unwrapped.state
    expected_reward = np.cos(next_state[2]) * 2.0
    np.testing.assert_allclose(custom_reward_value, expected_reward)


def test_custom_reward_with_different_obs_modes():
    """Test that custom reward works with different observation modes."""
    # Define a custom reward function that uses all parameters
    def complex_reward(state, action, next_state):
        prev_x, prev_x_dot, prev_theta, prev_theta_dot = state
        force = action[0]
        x, x_dot, theta, theta_dot = next_state
        
        # Calculate angle improvement (reward for getting closer to upright)
        angle_improvement = abs(prev_theta - np.pi) - abs(theta - np.pi)
        
        # Penalties
        action_penalty = -0.1 * abs(force)
        position_penalty = -0.05 * abs(x)
        
        return angle_improvement + action_penalty + position_penalty

    # Test with raw observation mode
    env_raw = gym.make("CartPoleSwingUp-v0", 
                      obs_mode="raw", 
                      custom_reward_fn=complex_reward)
    env_raw.reset(seed=42)
    
    # Test with trigonometric observation mode
    env_trig = gym.make("CartPoleSwingUp-v0", 
                       obs_mode="trig", 
                       custom_reward_fn=complex_reward)
    env_trig.reset(seed=42)
    
    # Apply same action to both environments
    action = np.array([0.5])
    _, reward_raw, _, _, _ = env_raw.step(action)
    _, reward_trig, _, _, _ = env_trig.step(action)
    
    # Even though the observation modes are different, the rewards should be identical
    # since the custom reward function receives the internal state representation
    np.testing.assert_allclose(reward_raw, reward_trig)


def test_reward_function_receives_correct_values():
    """Test that the custom reward function receives the correct state values."""
    # Create a reward function that will store the values it receives
    received_values = {"state": None, "action": None, "next_state": None}
    
    def recording_reward(state, action, next_state):
        # Record the values
        received_values["state"] = state
        received_values["action"] = action
        received_values["next_state"] = next_state
        return 1.0  # Return a constant reward
    
    # Create environment with this reward function
    env = gym.make("CartPoleSwingUp-v0", custom_reward_fn=recording_reward)
    env.reset(seed=42)
    
    # Store the state before stepping
    prev_state = env.unwrapped.state
    
    # Take an action
    action = np.array([0.75])
    _, _, _, _, _ = env.step(action)
    
    # Get the state after stepping
    next_state = env.unwrapped.state
    
    # Verify that the reward function received the correct values
    np.testing.assert_allclose(received_values["state"], prev_state)
    np.testing.assert_allclose(received_values["action"], action)
    np.testing.assert_allclose(received_values["next_state"], next_state)
