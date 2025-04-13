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
    
    Args:
        render_mode (str): Rendering mode ('human' or 'rgb_array').
        gravity (float): Gravitational acceleration.
        cart_mass (float): Mass of the cart.
        pole_mass (float): Mass of the pole.
        pole_length (float): Length of the pole.
        force_mag (float): Maximum force magnitude applied to the cart.
        dt (float): Simulation time step.
        friction (float): Friction coefficient.
        x_threshold (float): Threshold for cart position (episode terminates if exceeded).
        time_limit (int): Maximum number of steps per episode.
        cost_mode (str): Reward function mode ('default' or 'pilco').
        sigma_c (float): Parameter for PILCO reward function.
        obs_mode (str): Observation mode ('raw' or 'trig').
        custom_reward_fn (callable): Custom reward function.
        initial_state_mean (np.ndarray): Mean of the initial state distribution [x, x_dot, theta, theta_dot].
            Default is [0.0, 0.0, π, 0.0] (pole pointing down).
        initial_state_noise (np.ndarray): Standard deviation for each state component.
            Default is [0.05, 0.05, 0.05, 0.05].
    
    Note:
        The reset method can be used in two ways:
        1. Deterministic: Set exact initial state by passing 'initial_state' in options.
           Example: `env.reset(options={"initial_state": [0.0, 0.0, np.pi, 0.0]})`
        
        2. Default randomized: Use the environment's default initialization parameters.
           Example: `env.reset()`
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self, 
        render_mode: str = None,
        gravity: float = 9.82,
        cart_mass: float = 0.5,
        pole_mass: float = 0.5,
        pole_length: float = 0.6,
        force_mag: float = 10.0,
        dt: float = 0.1,
        friction: float = 0.1,
        x_threshold: float = 2.4,
        time_limit: int = 1000,
        cost_mode: str = "default",
        sigma_c: float = 0.25,
        obs_mode: str = "raw",
        custom_reward_fn: callable = None,
        initial_state_mean: np.ndarray = None,
        initial_state_noise: np.ndarray = None,
    ):
        super().__init__()
        # Physical constants and parameters
        self.g = gravity  # Gravitational acceleration
        self.m_c = cart_mass  # Cart mass
        self.m_p = pole_mass  # Pole mass
        self.total_m = self.m_c + self.m_p
        self.l = pole_length  # Pole length (meters)
        self.m_p_l = self.m_p * self.l
        self.force_mag = force_mag  # Force magnitude scale applied to cart
        self.dt = dt  # Simulation time step
        self.b = friction  # Friction coefficient
        self.t = 0  # Time steps counter
        self.t_limit = time_limit  # Episode step limit
        self.cost_mode = cost_mode
        self.sigma_c = sigma_c
        self.obs_mode = obs_mode  # Observation mode: 'raw' or 'trig'
        self.custom_reward_fn = custom_reward_fn  # Custom reward function
        
        # Initial state configuration
        # Default initial state: [x=0, x_dot=0, theta=pi, theta_dot=0]
        self.initial_state_mean = initial_state_mean if initial_state_mean is not None else np.array([0.0, 0.0, np.pi, 0.0], dtype=np.float32)
        # Default noise scale: [0.05, 0.05, 0.05, 0.05]
        self.initial_state_noise = initial_state_noise if initial_state_noise is not None else np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float32)

        # Failure thresholds
        self.theta_threshold_radians = (
            12 * 2 * math.pi / 360
        )  # 12 degrees (not used in this env)
        self.x_threshold = x_threshold  # Cart position limit (left/right boundary)

        # Action space: Force applied to cart (continuous value from -1.0 to 1.0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space depends on obs_mode
        if self.obs_mode == "raw":
            # Original: [x, x_dot, theta, theta_dot]
            high = np.array([
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                np.pi * 2,
                np.finfo(np.float32).max
            ], dtype=np.float32)
            self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        elif self.obs_mode == "trig":
            # Trigonometric: [x, x_dot, sin(theta), cos(theta), theta_dot]
            high = np.array([
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                1.0,  # sin(theta)
                1.0,  # cos(theta)
                np.finfo(np.float32).max
            ], dtype=np.float32)
            low = np.array([
                -self.x_threshold * 2,
                -np.finfo(np.float32).max,
                -1.0,  # sin(theta)
                -1.0,  # cos(theta)
                -np.finfo(np.float32).max
            ], dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            raise ValueError(f"Invalid obs_mode: {self.obs_mode}. Must be 'raw' or 'trig'")

        # Rendering related
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Initialize internal state
        self.state = None

    def reset(self, seed=None, options=None):
        # Reset using Gymnasium's convention: set seed and sample initial state
        super().reset(seed=seed)
        
        # Check if options contains exact initial state specification
        if options is not None and "initial_state" in options:
            # Set exact initial state (deterministic)
            self.state = np.array(options["initial_state"], dtype=np.float32)
        else:
            # Default: Use instance variables for randomization
            self.state = self.np_random.normal(
                loc=self.initial_state_mean,
                scale=self.initial_state_noise,
            )
            
        self.t = 0  # Reset step counter

        # Return observation based on the selected mode
        obs = self._get_obs()

        # Initialize screen when in human rendering mode
        if self.render_mode == "human":
            self.render()  # Render initial state

        return obs, {}

    def _get_obs(self):
        """Convert the internal state to the desired observation format."""
        x, x_dot, theta, theta_dot = self.state
        
        if self.obs_mode == "raw":
            # Return raw state [x, x_dot, theta, theta_dot]
            return np.array(self.state, dtype=np.float32)
        elif self.obs_mode == "trig":
            # Return [x, x_dot, sin(theta), cos(theta), theta_dot]
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            return np.array([x, x_dot, sin_theta, cos_theta, theta_dot], dtype=np.float32)
        else:
            raise ValueError(f"Invalid obs_mode: {self.obs_mode}")

    def _compute_default_reward(self, state):
        """Calculate the default reward function (cos(theta) * cos(x))."""
        x, _, theta, _ = state
        return math.cos(theta) * math.cos(x)
    
    def _compute_pilco_reward(self, state):
        """Calculate the PILCO reward function based on tip position."""
        x, _, theta, _ = state
        tip_x = x + self.l * math.sin(theta)
        tip_y = self.l * math.cos(theta)
        target_x = 0.0
        target_y = self.l
        square_distance = (tip_x - target_x)**2 + (tip_y - target_y)**2
        cost = 1 - math.exp(-square_distance / (2 * self.sigma_c**2))
        return -cost

    def step(self, action):
        # Convert action to force
        act = np.clip(action, -1.0, 1.0).astype(np.float32)
        # action is a shape=(1,) array, convert to scalar
        force = float(act[0] * self.force_mag)

        # Store previous state for reward calculation
        prev_state = self.state
        
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
        
        # Keep theta within [-pi, pi]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        self.state = (x, x_dot, theta, theta_dot)

        # Calculate reward
        if self.custom_reward_fn is not None:
            # Use custom reward function if provided
            # The function should take (state, action, next_state) as input
            reward = self.custom_reward_fn(prev_state, action, self.state)
        elif self.cost_mode == "pilco":
            reward = self._compute_pilco_reward(self.state)
        elif self.cost_mode == "default":
            reward = self._compute_default_reward(self.state)
        else:
            raise ValueError(f"Invalid cost_mode: {self.cost_mode}")

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

        # Return observation based on selected mode
        obs = self._get_obs()

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
            vec = pygame.math.Vector2(coord).rotate_rad(theta)
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
        pole_top = pygame.math.Vector2(0, pole_len).rotate_rad(theta)
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
