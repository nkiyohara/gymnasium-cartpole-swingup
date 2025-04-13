#!/usr/bin/env python3
"""
CartPole SwingUp Keyboard Control Example

Controls:
- Left Arrow: Apply force to the left (-1)
- Right Arrow: Apply force to the right (+1)
- Space: No force (0)
- R: Reset environment
- Q: Quit

This script allows you to manually control the CartPole SwingUp environment
using keyboard inputs to better understand the dynamics of the system.
"""

import sys
import time
import gymnasium as gym
import numpy as np
import pygame

# Import the cartpole_swingup environment
import gymnasium_cartpole_swingup

# Initialize pygame for keyboard capture
pygame.init()

# Create environment with lower FPS for better visibility
env = gym.make(
    "CartPoleSwingUp-v0",
    render_mode="human",
    # You can modify these parameters as needed
    gravity=9.82,
    cart_mass=0.5,
    pole_mass=0.5, 
    pole_length=0.6,
    force_mag=10.0,
    dt=0.01,  # Much smaller time step for accurate physics simulation
    friction=0.3,  # Increase friction to prevent energy buildup
)
# Hack to modify the render FPS
env.metadata["render_fps"] = 60

# Set up separate info display
info_display = pygame.display.set_mode((600, 100), pygame.RESIZABLE)
pygame.display.set_caption("Control Panel")
font = pygame.font.Font(None, 30)

# Create a clock for controlling the frame rate
clock = pygame.time.Clock()

def reset_env():
    """Reset the environment with a stable initial state"""
    # Use the reset with options to set an exact initial state
    # This makes it perfectly still at the beginning
    initial_state = np.array([0.0, 0.0, np.pi, 0.0], dtype=np.float32)  # x, x_dot, theta, theta_dot
    obs, _ = env.reset(options={"initial_state": initial_state})
    return obs

def handle_input():
    """Process keyboard input and return the corresponding action"""
    action = np.array([0.0])  # Default: no force
    should_quit = False
    should_reset = False
    
    # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            should_quit = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                should_quit = True
            elif event.key == pygame.K_r:
                should_reset = True
            elif event.key == pygame.K_SPACE:
                # Explicitly handle space key as zero force
                action = np.array([0.0])
                return action
    
    # Get pressed keys for continuous control
    keys = pygame.key.get_pressed()
    
    # Determine action based on key press
    if keys[pygame.K_LEFT]:
        action = np.array([-1.0])  # Left force
    elif keys[pygame.K_RIGHT]:
        action = np.array([1.0])   # Right force
    elif keys[pygame.K_SPACE]:
        action = np.array([0.0])   # Explicitly zero force when space is pressed
    
    if should_quit:
        return None
    
    if should_reset:
        print("Environment reset")
        return "RESET"
    
    return action

def display_info(obs, action, reward):
    """Display information about the current state"""
    # Clear the info display
    info_display.fill((255, 255, 255))
    
    # Format state information
    if len(obs) == 4:  # Raw observation mode
        x, x_dot, theta, theta_dot = obs
        state_text = f"x: {x:.2f}, x_dot: {x_dot:.2f}, θ: {theta:.2f}, θ_dot: {theta_dot:.2f}"
        action_text = f"Action: {action[0]:.1f}, Reward: {reward:.2f}"
    else:  # Trig observation mode
        x, x_dot, sin_theta, cos_theta, theta_dot = obs
        theta = np.arctan2(sin_theta, cos_theta)
        state_text = f"x: {x:.2f}, x_dot: {x_dot:.2f}, θ: {theta:.2f}, θ_dot: {theta_dot:.2f}"
        action_text = f"Action: {action[0]:.1f}, Reward: {reward:.2f}"
    
    # Display the text
    text_surface = font.render(state_text, True, (0, 0, 0))
    info_display.blit(text_surface, (10, 40))
    
    action_surface = font.render(action_text, True, (0, 0, 0))
    info_display.blit(action_surface, (10, 70))
    
    # Control instructions
    controls = "Controls: ←/→ = Force, Space = No force, R = Reset, Q = Quit"
    controls_surface = font.render(controls, True, (0, 0, 0))
    info_display.blit(controls_surface, (10, 10))
    
    # Update only the info display
    pygame.display.flip()

def main():
    """Main control loop"""
    # Initialize environment
    obs = reset_env()
    reward = 0
    action = np.array([0.0])
    
    print("CartPole SwingUp Keyboard Control Started")
    print("Controls:")
    print("  Left Arrow: Apply force to the left")
    print("  Right Arrow: Apply force to the right")
    print("  Space: No force (explicitly zero)")
    print("  R: Reset environment")
    print("  Q: Quit")
    
    # Physics and render settings
    physics_steps_per_frame = 5  # Run multiple physics steps per frame
    frame_time = 1/30  # Target 30 fps for display
    last_render_time = time.time()
    
    # Control loop
    running = True
    while running:
        # Limit frame rate for display
        clock.tick(30)  # 30 fps max for display
        
        # Get action from keyboard
        new_action = handle_input()
        
        # Check for quit signal
        if new_action is None:
            running = False
            break
        
        # Handle reset
        if new_action == "RESET":
            obs = reset_env()
            reward = 0
            action = np.array([0.0])
            last_render_time = time.time()
            continue
        
        # Update action if provided
        action = new_action
        
        # Perform multiple physics steps per frame
        for _ in range(physics_steps_per_frame):
            # Step environment (physics simulation)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # Check if episode is done
            if terminated or truncated:
                print("Episode ended. Resetting...")
                obs = reset_env()
                reward = 0
                break
        
        # Display state information and render only on display frame rate
        current_time = time.time()
        if current_time - last_render_time >= frame_time:
            display_info(obs, action, reward)
            last_render_time = current_time
    
    # Clean up
    env.close()
    pygame.quit()
    print("Exiting...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting (Ctrl+C pressed)")
        try:
            env.close()
            pygame.quit()
        except:
            pass
        sys.exit(0) 