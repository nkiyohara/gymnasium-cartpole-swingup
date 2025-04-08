import math

import gymnasium as gym
import numpy as np
import pygame  # Pygame is used for rendering (initialized when needed)
import pygame.gfxdraw
from gymnasium import spaces


class CartPoleSwingUpEnv(gym.Env):
    """
    Cart-pole swing-up environment.

    This environment is a modified version of the classic cart-pole, where the pole
    starts in a downward position and must be swung up and balanced.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str = None):
        super().__init__()
        # Physical constants and parameters
        self.g = 9.82  # Gravitational acceleration
        self.m_c = 0.5  # Cart mass
        self.m_p = 0.5  # Pole mass
        self.total_m = self.m_c + self.m_p
        self.l = 0.6  # Pole length (meters)
        self.m_p_l = self.m_p * self.l
        self.force_mag = 10.0  # Force magnitude scale applied to cart
        self.dt = 0.01  # Simulation time step
        self.b = 0.1  # Friction coefficient
        self.t = 0  # Time steps counter
        self.t_limit = 1000  # Episode step limit

        # Failure thresholds
        self.theta_threshold_radians = (
            12 * 2 * math.pi / 360
        )  # 12 degrees (not used in this env)
        self.x_threshold = 2.4  # Cart position limit (left/right boundary)

        # Action space: Force applied to cart (continuous value from -1.0 to 1.0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Observation space: [x, x_dot, cos(theta), sin(theta), theta_dot]
        high = np.array([np.finfo(np.float32).max] * 5, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Rendering related
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Initialize internal state
        self.state = None

    def reset(self, seed=None, options=None):
        # Reset using Gymnasium's convention: set seed and sample initial state
        super().reset(seed=seed)
        # Generate initial state with random noise (near x=0, theta=pi)
        # Using self.np_random instead of np.random for seed-controlled randomness
        self.state = self.np_random.normal(
            loc=np.array([0.0, 0.0, np.pi, 0.0], dtype=np.float32),
            scale=np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32),
        )
        self.t = 0  # Reset step counter

        # Calculate observation (x, x_dot, cosθ, sinθ, θ_dot)
        x, x_dot, theta, theta_dot = self.state
        obs = np.array(
            [x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32
        )

        # Initialize screen when in human rendering mode
        if self.render_mode == "human":
            self.render()  # Render initial state

        return obs, {}

    def step(self, action):
        # Convert action to force
        act = np.clip(action, -1.0, 1.0).astype(np.float32)
        # action is a shape=(1,) array, convert to scalar
        force = float(act[0] * self.force_mag)

        # Unpack state variables
        x, x_dot, theta, theta_dot = self.state
        # Calculate trigonometric functions
        s = math.sin(theta)
        c = math.cos(theta)
        # Update state based on analytical solution (CartPole dynamics equations)
        xdot_update = (
            -2 * self.m_p_l * (theta_dot**2) * s
            + 3 * self.m_p * self.g * s * c
            + 4 * force
            - 4 * self.b * x_dot
        ) / (4 * self.total_m - 3 * self.m_p * c**2)
        thetadot_update = (
            -3 * self.m_p_l * (theta_dot**2) * s * c
            + 6 * self.total_m * self.g * s
            + 6 * (force - self.b * x_dot) * c
        ) / (4 * self.l * self.total_m - 3 * self.m_p_l * c**2)
        # Update state using Euler method
        x = x + x_dot * self.dt
        theta = theta + theta_dot * self.dt
        x_dot = x_dot + xdot_update * self.dt
        theta_dot = theta_dot + thetadot_update * self.dt
        self.state = (x, x_dot, theta, theta_dot)

        # Calculate reward: higher when pole is upright (cosθ=1) and cart is near center
        reward_theta = (
            math.cos(theta) + 1.0
        ) / 2.0  # max 1 when cosθ=1, min 0 when cosθ=-1
        reward_x = math.cos(
            (x / self.x_threshold) * (math.pi / 2.0)
        )  # 1 at x=0, 0 at x=±x_threshold
        reward = reward_theta * reward_x

        # Termination conditions
        terminated = False
        truncated = False
        # Terminate if cart moves beyond boundaries (failure)
        if x < -self.x_threshold or x > self.x_threshold:
            terminated = True
        # Truncate if episode exceeds step limit
        self.t += 1
        if self.t >= self.t_limit:
            truncated = True
        # If both conditions are met, prioritize termination
        if terminated:
            truncated = False

        # Prepare next observation
        obs = np.array(
            [x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32
        )

        # Update rendering if in human mode
        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            # No rendering mode
            return None

        # Setup Pygame if not initialized
        if self.screen is None:
            # Initialize Pygame
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                # Create window
                self.screen = pygame.display.set_mode((600, 600))
            else:  # "rgb_array"
                # Offscreen surface for rendering
                self.screen = pygame.Surface((600, 600))
            self.clock = pygame.time.Clock()

        # Don't render if state is missing
        if self.state is None:
            return None

        # Clear screen with white background
        surf = pygame.Surface((600, 600))
        surf.fill((255, 255, 255))
        screen_width, screen_height = 600, 600

        # Scale for coordinate conversion (world width 5m maps to 600px)
        world_width = 5.0  # Visible range [-2.5, 2.5]m
        scale = screen_width / world_width
        # Cart drawing size
        carty = (
            screen_height / 2
        )  # Y-position of cart's top surface in drawing coordinates
        cart_width = 40.0
        cart_height = 20.0
        pole_width = 6.0
        pole_len = scale * self.l  # Pole length in pixels

        x, x_dot, theta, theta_dot = self.state
        # Cart center coordinates (px). x=0 corresponds to screen center
        cartx = x * scale + screen_width / 2.0

        # Draw cart (rectangle)
        left = -cart_width / 2
        right = cart_width / 2
        top = cart_height / 2
        bottom = -cart_height / 2
        cart_coords = [(left, bottom), (left, top), (right, top), (right, bottom)]
        # Offset to cart position
        cart_coords = [(cartx + cx, carty + cy) for (cx, cy) in cart_coords]
        # Draw anti-aliased polygon (red cart)
        pygame.gfxdraw.aapolygon(surf, cart_coords, (255, 0, 0))
        pygame.gfxdraw.filled_polygon(surf, cart_coords, (255, 0, 0))

        # Draw pole (rotated rectangle)
        # Pole endpoints (unrotated). Pivot is pole bottom (cart connection) = (0,0)
        left = -pole_width / 2
        right = pole_width / 2
        top = pole_len - pole_width / 2
        bottom = -pole_width / 2
        pole_coords = [(left, bottom), (left, top), (right, top), (right, bottom)]
        # Rotate pole by theta (negate since in pygame coordinate system, down is +θ)
        pole_coords_rotated = []
        for coord in pole_coords:
            vec = pygame.math.Vector2(coord).rotate_rad(-theta)
            pole_coords_rotated.append((vec[0] + cartx, vec[1] + carty))
        # Draw pole (blue)
        pygame.gfxdraw.aapolygon(surf, pole_coords_rotated, (0, 0, 255))
        pygame.gfxdraw.filled_polygon(surf, pole_coords_rotated, (0, 0, 255))

        # Draw axle (circle at pole base)
        axle_radius = pole_width / 2
        pygame.gfxdraw.aacircle(
            surf, int(cartx), int(carty), int(axle_radius), (26, 255, 255)
        )
        pygame.gfxdraw.filled_circle(
            surf, int(cartx), int(carty), int(axle_radius), (26, 255, 255)
        )

        # Draw pole tip (small black circle)
        # Use the same coordinate system and rotation as the pole itself
        # This ensures the black circle stays exactly at the end of the pole
        pole_top = pygame.math.Vector2(0, pole_len).rotate_rad(-theta)
        pole_top_x = int(cartx + pole_top.x)
        pole_top_y = int(carty + pole_top.y)

        pygame.gfxdraw.aacircle(
            surf, pole_top_x, pole_top_y, int(axle_radius), (0, 0, 0)
        )
        pygame.gfxdraw.filled_circle(
            surf, pole_top_x, pole_top_y, int(axle_radius), (0, 0, 0)
        )

        # Draw wheels (black circles on left/right bottom of cart)
        wheel_radius = cart_height / 4.0
        # Left wheel center coordinates
        wheel_x_left = cartx - cart_width / 2
        wheel_y = carty - cart_height / 2
        # Right wheel center coordinates
        wheel_x_right = cartx + cart_width / 2
        # Draw left and right wheels
        pygame.gfxdraw.aacircle(
            surf, int(wheel_x_left), int(wheel_y), int(wheel_radius), (0, 0, 0)
        )
        pygame.gfxdraw.filled_circle(
            surf, int(wheel_x_left), int(wheel_y), int(wheel_radius), (0, 0, 0)
        )
        pygame.gfxdraw.aacircle(
            surf, int(wheel_x_right), int(wheel_y), int(wheel_radius), (0, 0, 0)
        )
        pygame.gfxdraw.filled_circle(
            surf, int(wheel_x_right), int(wheel_y), int(wheel_radius), (0, 0, 0)
        )

        # Draw track ground line (black line)
        track_y = carty - cart_height / 2 - wheel_radius  # Ground height
        pygame.gfxdraw.hline(surf, 0, screen_width, int(track_y), (0, 0, 0))

        # Flip the rendered surface vertically to apply to screen (convert coordinate system upward)
        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        if self.render_mode == "human":
            # Display on screen
            pygame.display.flip()
            # Wait to maintain appropriate FPS
            self.clock.tick(self.metadata["render_fps"])
            # Process event loop (to keep window responsive)
            pygame.event.pump()
            return None
        elif self.render_mode == "rgb_array":
            # Return pixel array
            image = pygame.surfarray.pixels3d(self.screen)
            return np.transpose(np.array(image), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            # Release Pygame resources
            try:
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
            self.screen = None
            self.clock = None
