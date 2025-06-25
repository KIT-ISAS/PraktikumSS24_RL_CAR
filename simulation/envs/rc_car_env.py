import gymnasium as gym
import numpy as np

class RCCarNavigationEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.dt = 0.1
        self.L = 0.25
        self.max_speed = 1.0
        self.max_steering = 0.5
        self.area_size = 4.0
        self.max_episode_steps = 1500

        self.goal_hold_duration = 50  # 2 Sekunden bei dt=0.1
        self.goal_hold_timer = 0
        self.goal_reached = False

        self.steering_hold_counter = 0
        self.prev_steering = 0.0

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=np.array([-self.max_speed, -self.max_steering]),
            high=np.array([self.max_speed, self.max_steering]),
            dtype=np.float32
        )

        self.reset()

    def reset(self, *, seed=None, options=None, set_new_goal=True):
        super().reset(seed=seed)
        self.position = np.array([0.0, 0.0])
        self.theta = 0.0
        self.v = 0.0
        self.last_steering = 0.0
        self.prev_theta = self.theta
        self.prev_position = self.position.copy()
        self.prev_dist = None
        self.prev_v_sign = 1
        self.steps = 0
        self.v_switches = 0
        self.prev_steering = 0.0
        self.steering_hold_counter = 0

        self.goal_hold_timer = 0
        self.goal_reached = False

        if set_new_goal:
            self.goal = self._sample_goal()

        return self._get_obs(), {}

    def step(self, action):
        speed, steering = action
        speed = np.clip(speed, -self.max_speed, self.max_speed)
        steering = np.clip(steering, -self.max_steering, self.max_steering)
        self.v = speed
        self.last_steering = steering

        # Fahrzeugkinematik
        x, y = self.position
        x += self.v * np.cos(self.theta) * self.dt
        y += self.v * np.sin(self.theta) * self.dt
        self.theta += (self.v / self.L) * np.tan(steering) * self.dt
        self.position = np.array([x, y])

        obs = self._get_obs()
        reward = self._compute_reward()

        # === Zielverhalten ===
        dist_to_goal = np.linalg.norm(self.goal - self.position)
        goal_radius = 0.2
        goal_reached_now = False

        if dist_to_goal < goal_radius:
            self.goal_hold_timer += 1
            if self.goal_hold_timer == self.goal_hold_duration:
                reward += 100.0  # Ziel erfolgreich erreicht
                goal_reached_now = True
                self.goal = self._sample_goal()
                self.goal_hold_timer = 0
        else:
            self.goal_hold_timer = 0

        # === Rand überschreiten – sehr starke Strafe ===
        out_of_bounds = np.any(np.abs(self.position) > self.area_size / 2)
        if out_of_bounds:
            reward -= 100.0
            truncated = True
        else:
            truncated = self.steps >= self.max_episode_steps

        # === Lenkwinkel-Halte-Strafe ===
        if abs(steering - self.prev_steering) < 0.01:
            self.steering_hold_counter += 1
        else:
            self.steering_hold_counter = 0

        if self.steering_hold_counter > 30:
            reward -= 2.0  # Strafe für zu lang gleiches Lenken

        self.prev_steering = steering

        terminated = False
        self.steps += 1
        return obs, reward, terminated, truncated, {"goal_reached": goal_reached_now}

    def _get_obs(self):
        dx, dy = self.goal - self.position
        distance = np.linalg.norm([dx, dy])
        angle_to_goal = np.arctan2(dy, dx)
        rel_angle = angle_to_goal - self.theta
        rel_angle = np.arctan2(np.sin(rel_angle), np.cos(rel_angle))  # normieren

        return np.array([
            dx, dy,
            distance,
            np.sin(rel_angle), np.cos(rel_angle),
            self.v
        ], dtype=np.float32)

    def _compute_reward(self):
        dist_to_goal = np.linalg.norm(self.goal - self.position)
        reward = -dist_to_goal

        if self.prev_dist is not None:
            delta = self.prev_dist - dist_to_goal
            if delta > 0.05:
                reward += delta * 10.0
        self.prev_dist = dist_to_goal

        if abs(self.v) < 0.01 and dist_to_goal > 0.2:
            reward -= 1.5

        if self.v < -0.1:
            reward -= 1.0
        elif self.v > 0.1:
            reward += 1.0

        if dist_to_goal < 0.2 and abs(self.v) < 0.1:
            reward += 5.0  # ruhig im Ziel stehen

        if not hasattr(self, "v_switches"):
            self.v_switches = 0

        current_v_sign = np.sign(self.v)
        if current_v_sign != self.prev_v_sign and abs(self.v) > 0.1:
            self.v_switches += 1
            reward -= 5.0  # starke Bestrafung pro Wechsel

        self.prev_v_sign = current_v_sign

        # Optional: zusätzliche Strafe bei extrem häufigem Wechseln
        if self.v_switches >= 5:
            reward -= 10.0  # nochmalige Zusatzstrafe
            self.v_switches = 0  # zurücksetzen

        angle_diff = abs(self.theta - self.prev_theta)
        move_diff = np.linalg.norm(self.position - self.prev_position)
        if angle_diff > 0.3 and move_diff < 0.05:
            reward -= 3.0

        self.prev_theta = self.theta
        self.prev_position = self.position.copy()

        return reward

    def _sample_goal(self):
        while True:
            goal = np.random.uniform(-self.area_size/2 + 0.5, self.area_size/2 - 0.5, size=2)
            if np.linalg.norm(goal) > 1.0:
                return goal
