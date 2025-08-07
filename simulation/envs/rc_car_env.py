"""
rc_car_env.py

Gymnasium environment for RC car navigation.

This environment simulates a differential-drive RC car in a square arena
with goal-reaching tasks. The state space includes relative goal position,
heading information, and speed. Actions are continuous speed and steering
commands. Rewards are shaped based on distance to goal, progress, and
various penalties for undesirable behaviors.
"""

import gymnasium as gym
import numpy as np

class RCCarNavigationEnv(gym.Env):
    """
    Custom Gymnasium environment for RC car navigation.

    State (6-dimensional):
      0: dx (goal x - position x)
      1: dy (goal y - position y)
      2: distance to goal
      3: sin(relative angle to goal)
      4: cos(relative angle to goal)
      5: current forward speed v

    Action (2-dimensional):
      0: speed command in [-max_speed, max_speed]
      1: steering command in [-max_steering, max_steering]

    The car follows a simple kinematic bicycle model. Episodes terminate
    when the car goes out of bounds or after a fixed number of steps.
    A large bonus is given for holding at the goal for a specified duration.
    """
    def __init__(self):
        super().__init__()
        # Simulation parameters
        self.dt = 0.1                     # Time step (seconds)
        self.L = 0.25                     # Wheelbase length (meters)
        self.max_speed = 1.0             # Maximum forward/backward speed
        self.max_steering = 0.5          # Maximum steering angle (radians)
        self.area_size = 4.0             # Side length of square arena
        self.max_episode_steps = 1500    # Episode length limit

        # Goal holding parameters
        self.goal_hold_duration = 50     # Steps to hold at goal (2 seconds)
        self.goal_hold_timer = 0         # Counter for hold duration
        self.goal_reached = False        # Flag for goal achieved event

        # Steering-hold penalty tracking
        self.steering_hold_counter = 0
        self.prev_steering = 0.0

        # Define observation and action spaces
        high_obs = np.array([np.inf]*6, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-high_obs, high=high_obs, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-self.max_speed, -self.max_steering], dtype=np.float32),
            high=np.array([ self.max_speed,  self.max_steering], dtype=np.float32),
            dtype=np.float32
        )

        # Initialize state
        self.reset()

    def reset(self, *, seed=None, options=None, set_new_goal=True):
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility
            set_new_goal: If True, sample a new goal position
        Returns:
            observation: initial state vector
            info: empty dict (per Gymnasium API)
        """
        super().reset(seed=seed)
        # Car state
        self.position = np.array([0.0, 0.0])
        self.theta = 0.0             # Heading angle
        self.v = 0.0                 # Forward speed
        self.last_steering = 0.0     # Most recent steering angle
        # For reward shaping and penalties
        self.prev_theta = self.theta
        self.prev_position = self.position.copy()
        self.prev_dist = None
        self.prev_v_sign = 1
        self.v_switches = 0
        self.prev_steering = 0.0
        self.steering_hold_counter = 0
        # Episode counters
        self.steps = 0
        self.goal_hold_timer = 0
        self.goal_reached = False
        # Sample a new goal if requested
        if set_new_goal:
            self.goal = self._sample_goal()
        # Return initial observation
        return self._get_obs(), {}

    def step(self, action):
        """
        Advance the simulation by one time step.

        Args:
            action: array-like [speed, steering]
        Returns:
            obs: next observation
            reward: float scalar reward
            terminated: bool (always False here)
            truncated: bool (True if out of bounds or max steps reached)
            info: dict containing 'goal_reached' flag
        """
        # Clip and unpack action
        speed, steering = action
        self.v = np.clip(speed, -self.max_speed, self.max_speed)
        self.last_steering = np.clip(steering, -self.max_steering, self.max_steering)

        # Kinematic update
        x, y = self.position
        x += self.v * np.cos(self.theta) * self.dt
        y += self.v * np.sin(self.theta) * self.dt
        self.theta += (self.v / self.L) * np.tan(self.last_steering) * self.dt
        self.position = np.array([x, y])

        # Compute observation and base reward
        obs = self._get_obs()
        reward = self._compute_reward()

        # Goal-reaching logic: hold within radius for bonus
        dist_to_goal = np.linalg.norm(self.goal - self.position)
        goal_radius = 0.2
        goal_reached_now = False
        if dist_to_goal < goal_radius:
            self.goal_hold_timer += 1
            if self.goal_hold_timer >= self.goal_hold_duration:
                reward += 100.0
                goal_reached_now = True
                self.goal = self._sample_goal()
                self.goal_hold_timer = 0
        else:
            self.goal_hold_timer = 0

        # Out-of-bounds or time truncation
        out_of_bounds = np.any(np.abs(self.position) > self.area_size/2)
        truncated = out_of_bounds or (self.steps >= self.max_episode_steps)
        if out_of_bounds:
            reward -= 100.0

        # Steering-hold penalty
        if abs(self.last_steering - self.prev_steering) < 0.01:
            self.steering_hold_counter += 1
        else:
            self.steering_hold_counter = 0
        if self.steering_hold_counter > 30:
            reward -= 2.0
        self.prev_steering = self.last_steering

        # Increment step counter
        self.steps += 1
        # Always continue; termination via truncated
        terminated = False
        # Return step tuple
        return obs, reward, terminated, truncated, {"goal_reached": goal_reached_now}

    def _get_obs(self):
        """
        Compute the current observation vector.
        Returns:
            obs: np.ndarray of shape (6,)
        """
        dx, dy = self.goal - self.position
        dist = np.linalg.norm([dx, dy])
        angle_to_goal = np.arctan2(dy, dx)
        rel_angle = angle_to_goal - self.theta
        rel_angle = np.arctan2(np.sin(rel_angle), np.cos(rel_angle))
        return np.array([dx, dy, dist,
                         np.sin(rel_angle), np.cos(rel_angle),
                         self.v], dtype=np.float32)

    def _compute_reward(self):
        """
        Internal reward shaping based on progress and penalties.
        Returns:
            reward: float scalar
        """
        dist = np.linalg.norm(self.goal - self.position)
        reward = -dist

        # Reward for forward progress
        if self.prev_dist is not None:
            delta = self.prev_dist - dist
            if delta > 0.05:
                reward += delta * 10.0
        self.prev_dist = dist

        # Penalty for standing still far from goal
        if abs(self.v) < 0.01 and dist > 0.2:
            reward -= 1.5

        # Small penalty or reward for backward/forward motion
        if self.v < -0.1:
            reward -= 1.0
        elif self.v > 0.1:
            reward += 1.0

        # Bonus for gentle hold at goal
        if dist < 0.2 and abs(self.v) < 0.1:
            reward += 5.0

        # Penalty for rapid speed reversals
        current_sign = np.sign(self.v)
        if current_sign != self.prev_v_sign and abs(self.v) > 0.1:
            self.v_switches += 1
            reward -= 5.0
        self.prev_v_sign = current_sign
        if self.v_switches >= 5:
            reward -= 10.0
            self.v_switches = 0

        # Penalty for turning in place without movement
        angle_diff = abs(self.theta - self.prev_theta)
        move_diff = np.linalg.norm(self.position - self.prev_position)
        if angle_diff > 0.3 and move_diff < 0.05:
            reward -= 3.0
        self.prev_theta = self.theta
        self.prev_position = self.position.copy()

        return reward

    def _sample_goal(self):
        """
        Sample a new goal position uniformly in the arena, excluding a central deadzone.
        Returns:
            goal: np.ndarray of shape (2,)
        """
        while True:
            goal = np.random.uniform(-self.area_size/2 + 0.5,
                                     self.area_size/2 - 0.5, size=2)
            if np.linalg.norm(goal) > 1.0:
                return goal
