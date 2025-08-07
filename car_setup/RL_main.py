"""
Main script for Reinforcement Learning (RL) training of a VR-tracked agent.

This script defines actor and critic networks, a replay buffer, reward computation,
client-server communication, training loop, and live visualization of training
metrics.
"""

# === Imports ===
import os
import time
import json
import socket
import random
import pickle
from collections import deque
from datetime import datetime
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import triad_openvr

# === Hyperparameters ===
LR_ACTOR = 1e-4           # Learning rate for the actor network
LR_CRITIC = 1e-3          # Learning rate for the critic network
GAMMA = 0.99              # Discount factor for future rewards
TAU = 0.005               # Soft update coefficient for target networks
BUFFER_SIZE = 1_000_000   # Maximum size of the replay buffer
BATCH_SIZE = 256          # Mini-batch size for training updates
TRAIN_INTERVAL = 10       # Number of iterations between training steps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Compute device

# === Field boundaries ===
FIELD_X_LIMITS = (-1.2, 1.2)  # Allowed range for X coordinate
FIELD_Y_LIMITS = (-1.5, 1.5)  # Allowed range for Y coordinate

# === Velocity deadzone threshold ===
SPEED_THRESHOLD = 0.85  # Minimum speed magnitude before penalty

# === Globals for reward tracking ===
no_move_start_time = None  # Timestamp when agent stops making progress
prev_speed_sign = 0        # Sign of previous speed for direction-switch penalty
steering_hold_counter = 0  # Counter for consecutive similar steering inputs
prev_steering = 0.0        # Previous steering angle
v_switch_counter = 0       # Counter for rapid speed direction changes


# === Actor Network ===
class Actor(nn.Module):
    """
    Policy network mapping states to actions.
    Architecture: [state_dim] -> 256 -> 256 -> [action_dim], with Tanh output.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward pass through the actor network.

        Args:
            x: Torch tensor of shape (batch_size, state_dim)

        Returns:
            Torch tensor of shape (batch_size, action_dim) with values in [-1, 1]
        """
        return self.net(x)


# === Critic Network ===
class Critic(nn.Module):
    """
    Value network estimating Q-values for state-action pairs.
    Architecture: [state_dim + action_dim] -> 256 -> 256 -> 1.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, s, a):
        """
        Forward pass through the critic network.

        Args:
            s: Torch tensor of states, shape (batch_size, state_dim)
            a: Torch tensor of actions, shape (batch_size, action_dim)

        Returns:
            Torch tensor of Q-values, shape (batch_size, 1)
        """
        return self.net(torch.cat([s, a], dim=1))


# === Replay Buffer ===
class ReplayBuffer:
    """
    Fixed-capacity buffer storing transitions for experience replay.
    Each transition: (state, action, reward, next_state, done_flag).
    """
    def __init__(self, capacity):
        """Initialize buffer with a maximum capacity."""
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        """
        Add a transition to the buffer, evicting oldest if full.

        Args:
            transition: Tuple(state, action, reward, next_state, done)
        """
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        Sample a random mini-batch of transitions.

        Returns:
            Tuple of torch tensors: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(a), dtype=torch.float32).to(DEVICE),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.tensor(np.array(s2), dtype=torch.float32).to(DEVICE),
            torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        )

    def save(self, filepath):
        """
        Serialize buffer to disk using pickle.

        Args:
            filepath: Destination file path for the pickle.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)
        print(f"ReplayBuffer saved to {filepath}")

    def load(self, filepath):
        """
        Load buffer contents from a pickle file if it exists.

        Args:
            filepath: Path to the saved buffer file.
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.buffer = pickle.load(f)
            print(f"ReplayBuffer loaded from {filepath}")

    def __len__(self):
        """Return current number of stored transitions."""
        return len(self.buffer)


# === Reward Function ===
def compute_reward(position, goal, prev_dist, speed, steering, yaw, holding=False):
    """
    Compute scalar reward for agent's state and action.

    Args:
        position: np.array([x, y]) current agent position
        goal:     np.array([x_goal, y_goal]) target position
        prev_dist: float distance from agent to goal in previous step
        speed:    float commanded speed action
        steering: float commanded steering action
        yaw:      float agent heading angle in degrees
        holding:  bool whether agent is holding at goal

    Returns:
        (reward, current_distance)
    """
    global no_move_start_time, steering_hold_counter, prev_steering
    global prev_speed_sign, v_switch_counter

    reward = 0.0
    dist = np.linalg.norm(goal - position)

    # 1. Holding at goal yields fixed high reward
    if holding:
        return 20.0, dist

    # 2. Reward for approaching goal (inverse distance)
    reward += 3.0 / max(dist, 0.1)

    # 3. Out-of-bounds penalty if leaving field limits
    if not (FIELD_X_LIMITS[0] <= position[0] <= FIELD_X_LIMITS[1]
            and FIELD_Y_LIMITS[0] <= position[1] <= FIELD_Y_LIMITS[1]):
        return -20.0, dist

    # 4. Reward or penalty based on change in distance to goal
    if prev_dist is not None:
        progress = prev_dist - dist
        reward += progress * 5.0 / max(dist, 0.1)
        if abs(progress) > 0.03:
            no_move_start_time = None
            if progress < -0.1:
                reward -= abs(progress) * 5.0
        else:
            if no_move_start_time is None:
                no_move_start_time = time.time()

    # 5. Penalty for standing still too long
    if no_move_start_time is not None:
        still_time = time.time() - no_move_start_time
        if still_time > 2:
            reward -= min((still_time - 20) * 0.1, 20.0)

    # 6. Deadzone penalty if too slow far from goal
    if dist > 0.3 and abs(speed) < 0.55:
        reward -= 5.0

    # 8. Penalty for rapid speed reversals
    current_sign = np.sign(speed)
    if current_sign != 0 and current_sign != prev_speed_sign:
        v_switch_counter += 1
        reward -= min(2.0 * v_switch_counter, 5.0)
        prev_speed_sign = current_sign
    else:
        v_switch_counter = max(0, v_switch_counter - 1)

    # 9. Penalty for circling (constant steering)
    if abs(steering - prev_steering) < 0.2:
        steering_hold_counter += 1
    else:
        steering_hold_counter = 0

    if steering_hold_counter > 30:
        reward -= 2.0 + (steering_hold_counter - 30) * 0.1

    prev_steering = steering

    # 10. Clip reward to avoid extreme values
    reward = np.clip(reward, -20.0, 20.0)
    return reward, dist


def reset_reward_tracking():
    """
    Reset global counters used by compute_reward().
    """
    global steering_hold_counter, prev_steering, v_switch_counter, prev_speed_sign
    steering_hold_counter = 0
    prev_steering = 0.0
    v_switch_counter = 0
    prev_speed_sign = 0


# === Client for VR Interaction and Training Loop ===
class Client:
    """
    Handles VR sensor input, environment interaction, RL training, and
    communication with external controller via TCP socket.
    """
    def __init__(self, host="192.168.137.5", port=12345):
        """
        Initialize VR interface, socket, networks, buffer, logs, and state.
        """
        self.vr = triad_openvr.triad_openvr()
        self.vr.print_discovered_objects()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

        self.actor = Actor(5, 2).to(DEVICE)
        self.critic = Critic(5, 2).to(DEVICE)
        self.actor_target = Actor(5, 2).to(DEVICE)
        self.critic_target = Critic(5, 2).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.replay_path = "models/replay_buffer.pkl"

        self.episode = 0
        self.iteration = 0
        self.loss_log = []
        self.prev_reward = 0.0

        self.goal = self._sample_goal()
        self.return_goal = np.array([0.0, 0.0])
        self.current_goal = self.goal
        self.current_position = np.array([0.0, 0.0])

        self.prev_obs = None
        self.prev_dist = None
        self.paused = False
        self.state = "orienting"
        self.goal_reached_time = None
        self.hold_timer_active = False
        self.out_of_bounds_handled = False

        self.prev_speed_sign = 0
        self.in_neutral_phase = False
        self.neutral_count = 0

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_log_filename = (
            f"plot_results/measurement_data/episode_{self.episode}_{timestamp_str}.json"
        )
        os.makedirs(os.path.dirname(self.data_log_filename), exist_ok=True)
        self.raw_data_log = []

    def _sample_goal(self):
        """
        Sample a new random goal position outside a central deadzone.

        Returns:
            NumPy array of shape (2,) with distance > 0.5 from origin.
        """
        while True:
            g = np.random.uniform(-1.0, 1.0, size=2)
            if np.linalg.norm(g) > 0.5:
                return g

    def load_models(self):
        """
        Load existing actor and critic models from disk, if available.
        """
        actor_path = "models/actor_latest.pt"
        critic_path = "models/critic_latest.pt"
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_state_dict(torch.load(actor_path))
            self.critic.load_state_dict(torch.load(critic_path))
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            print("Models loaded for continued training.")

    def save_models(self):
        """
        Save current actor and critic weights to disk (per-episode and latest).
        """
        os.makedirs("models", exist_ok=True)
        torch.save(self.actor.state_dict(), f"models/actor_ep{self.episode}.pt")
        torch.save(self.critic.state_dict(), f"models/critic_ep{self.episode}.pt")
        torch.save(self.actor.state_dict(), "models/actor_latest.pt")
        torch.save(self.critic.state_dict(), "models/critic_latest.pt")
        print("Models saved to disk.")

    def save_logs(self):
        """
        Persist training loss logs as JSON.
        """
        os.makedirs("plot_results/loss_logs", exist_ok=True)
        with open(f"plot_results/loss_logs/loss_log_ep{self.episode}.json", "w") as f:
            json.dump(self.loss_log, f, indent=2)
            print("Loss log saved.")

    def send_speed_message(self, speed, steering):
        """
        Send speed and steering commands to the connected controller,
        handling neutral-phase zeroing upon direction change.

        Args:
            speed: float command value
            steering: float command value
        """
        current_sign = np.sign(speed)
        if current_sign != 0 and current_sign != self.prev_speed_sign:
            print("Direction change detected - sending neutral signals.")
            self.in_neutral_phase = True
            self.neutral_count = 0
            self.prev_speed_sign = current_sign
        if self.in_neutral_phase:
            if self.neutral_count < 7:
                neutral = json.dumps({"speed": 0.0, "steering": 0.0}) + "\n"
                self.sock.sendall(neutral.encode())
                self.neutral_count += 1
                time.sleep(0.025)
                return
            else:
                self.in_neutral_phase = False
        msg = json.dumps({"speed": float(speed), "steering": float(steering)}) + "\n"
        self.sock.sendall(msg.encode())

    def send_loop(self):
        """
        Main control loop: read VR pose, compute action or return-to-center,
        send commands, collect experience, train networks, and handle logging.
        """
        try:
            self.goal_start_time = time.time()
            self.timer_no_move = time.time()
            self.goal_reached_time = None
            self.hold_timer_active = False
            prev_x = prev_y = prev_time = None

            while True:
                # Read pose from VR tracker
                pose = self.vr.devices["tracker_1"].get_pose_euler()
                x, y, yaw = pose[0], pose[1], pose[3]
                timestamp = time.time()
                position = np.array([x, y])
                self.current_position = position
                self.current_goal = self.goal

                # Compute instantaneous speed
                if prev_x is not None:
                    dx, dy = x - prev_x, y - prev_y
                    dt = timestamp - prev_time
                    speed_calculated = np.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0.0
                else:
                    speed_calculated = 0.0
                prev_x, prev_y, prev_time = x, y, timestamp

                # Handle out-of-bounds and reset
                if not self.paused and not self.out_of_bounds_handled and not (
                    FIELD_X_LIMITS[0] <= x <= FIELD_X_LIMITS[1] and
                    FIELD_Y_LIMITS[0] <= y <= FIELD_Y_LIMITS[1]
                ):
                    print("Out of bounds - applying penalty and returning to center.")
                    reward, _ = compute_reward(
                        position, self.goal, self.prev_dist,
                        speed_calculated, 0.0, yaw, holding=False
                    )
                    done = True
                    next_state = self.prev_obs if self.prev_obs is not None else None
                    if self.prev_obs is not None:
                        self.buffer.push((self.prev_obs, [0.0, 0.0], reward, next_state, 1.0))
                    self.loss_log.append({
                        "episode": self.episode,
                        "critic_loss": self.loss_log[-1]["critic_loss"] if self.loss_log else 0.0,
                        "actor_loss": self.loss_log[-1]["actor_loss"] if self.loss_log else 0.0,
                        "reward": reward,
                        "timestamp": time.time()
                    })
                    self.save_raw_data()
                    self.raw_data_log = []
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.data_log_filename = (
                        f"plot_results/measurement_data/episode_{self.episode}_{timestamp_str}.json"
                    )
                    self.prev_obs = None
                    self.prev_dist = None
                    self.goal_reached_time = None
                    self.hold_timer_active = False
                    self.save_models()
                    self.paused = True
                    self.state = "orienting"
                    self.out_of_bounds_handled = True
                    continue

                # Return-to-center logic when paused
                if self.paused:
                    reset_reward_tracking()
                    dx, dy = self.return_goal - position
                    distance = np.linalg.norm([dx, dy])
                    car_angle = (-yaw - 150) % 360
                    goal_angle = np.degrees(np.arctan2(dy, dx)) % 360
                    angle_diff = (goal_angle - car_angle + 540) % 360 - 180
                    orientation_threshold = 3
                    time.sleep(0.02)

                    if distance < 0.45:
                        speed_cmd, steer_cmd = 0.0, 0.0
                        self.paused = False
                        self.prev_obs = None
                        self.goal_start_time = time.time()
                        self.out_of_bounds_handled = False
                    else:
                        if self.state == "orienting":
                            if abs(angle_diff) < orientation_threshold:
                                speed_cmd, steer_cmd = 0.95, 0.0
                            else:
                                damb = angle_diff / 180.0
                                speed_cmd, steer_cmd = 1.0, -damb * 3.0
                    self.send_speed_message(speed_cmd, steer_cmd)
                    continue

                # RL state: compute distance, angle difference, normalized state
                dx, dy = self.goal - position
                dist = np.linalg.norm([dx, dy])
                car_angle = (-yaw - 150) % 360
                goal_angle = np.degrees(np.arctan2(dy, dx)) % 360
                angle_diff = (goal_angle - car_angle + 180) % 360 - 180
                yaw_norm = yaw / 180.0
                state = np.array([self.goal[0], self.goal[1], x, y, yaw_norm], dtype=np.float32)
                obs_tensor = torch.tensor(state).unsqueeze(0).to(DEVICE)

                # Select action (random vs policy + exploration)
                if len(self.buffer) < BATCH_SIZE:
                    action = np.random.uniform(-1, 1, size=2)
                else:
                    with torch.no_grad():
                        action = self.actor(obs_tensor).cpu().numpy()[0]
                        noise = max(0.05, 0.2 * np.exp(-self.episode / 300))
                        action += np.random.normal(0, noise, size=2)
                        action = np.clip(action, -1.0, 1.0)

                speed_cmd, steer_cmd = action
                reward, current_dist = compute_reward(
                    position, self.goal, self.prev_dist,
                    speed_cmd, steer_cmd, yaw
                )
                self.prev_dist = current_dist
                next_state = state.copy()
                done = current_dist < 0.2

                # Store transition if previous state exists
                if self.prev_obs is not None:
                    self.buffer.push((self.prev_obs, action, reward, next_state, float(done)))

                self.prev_obs = next_state
                self.prev_reward = reward

                # Append raw data log entry
                self.raw_data_log.append({
                    "timestamp": timestamp,
                    "episode": self.episode,
                    "position": {"x": x, "y": y},
                    "yaw": yaw,
                    "goal": {"x": float(self.goal[0]), "y": float(self.goal[1])},
                    "speed_calculated": speed_calculated,
                    "paused": self.paused,
                    "action": {"speed": float(speed_cmd), "steering": float(steer_cmd)},
                    "reward": float(reward),
                    "critic_loss": float(self.loss_log[-1]["critic_loss"]) if self.loss_log else None,
                    "actor_loss": float(self.loss_log[-1]["actor_loss"]) if self.loss_log else None
                })

                # Perform training step at specified interval
                if self.iteration % TRAIN_INTERVAL == 0 and len(self.buffer) >= BATCH_SIZE:
                    self.train_step()

                # Send control command to controller
                self.send_speed_message(speed_cmd, steer_cmd)
                print(f"Position: ({x:.2f}, {y:.2f}) | Goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f}) | Distance: {current_dist:.3f} | Done: {done}", end="\r", flush=True)

                # Handle goal reached and hold logic
                if done:
                    if self.goal_reached_time is None:
                        self.goal_reached_time = time.time()
                        self.hold_timer_active = True
                    elif time.time() - self.goal_reached_time >= 0.5:
                        print("Goal reached and held.")
                        reward = 10.0
                        if self.prev_obs is not None:
                            self.buffer.push((self.prev_obs, [0.0, 0.0], reward, next_state, 1.0))
                        self.loss_log.append({
                            "episode": self.episode,
                            "critic_loss": self.loss_log[-1]["critic_loss"] if self.loss_log else 0.0,
                            "actor_loss": self.loss_log[-1]["actor_loss"] if self.loss_log else 0.0,
                            "reward": reward,
                            "timestamp": time.time()
                        })
                        self.save_raw_data()
                        self.raw_data_log = []
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        self.data_log_filename = (
                            f"plot_results/measurement_data/episode_{self.episode}_{timestamp_str}.json"
                        )
                        self.goal = self._sample_goal()
                        self.prev_obs = None
                        self.prev_dist = None
                        self.episode += 1
                        self.goal_reached_time = None
                        self.hold_timer_active = False
                        self.save_models()
                        continue
                    else:
                        self.send_speed_message(0.0, 0.0)
                        reward, _ = compute_reward(
                            position, self.goal, self.prev_dist,
                            0.0, 0.0, yaw, holding=True
                        )
                        self.buffer.push((self.prev_obs, [0.0, 0.0], reward, next_state, 0.0))
                        time.sleep(0.005)
                        continue
                else:
                    self.goal_reached_time = None
                    self.hold_timer_active = False

                reset_reward_tracking()
                time.sleep(0.005)

        except KeyboardInterrupt:
            print("Training interrupted, shutting down.")
            self.sock.close()
            self.save_logs()
            self.save_raw_data()

    def train_step(self):
        """
        Sample a batch from replay buffer and update actor and critic networks.
        Perform soft updates of target networks.
        """
        s, a, r, s2, d = self.buffer.sample(BATCH_SIZE)
        with torch.no_grad():
            a2 = self.actor_target(s2)
            q_target = r + GAMMA * (1 - d) * self.critic_target(s2, a2)
        q_val = self.critic(s, a)
        critic_loss = nn.MSELoss()(q_val, q_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Soft update target networks
        for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        self.loss_log.append({
            "timestamp": time.time(),
            "episode": self.episode,
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "reward": self.prev_reward
        })

    def save_raw_data(self):
        """
        Persist raw movement and transition data, then save replay buffer.
        """
        with open(self.data_log_filename, "w") as f:
            json.dump(self.raw_data_log, f, indent=2)
        self.buffer.save(self.replay_path)
        print(f"Movement data saved to: {self.data_log_filename}")


# === Live Visualization ===
def start_live_plot(client_instance):
    """
    Launch a Matplotlib animation showing:
      - Agent position and goal in XY-plane
      - Critic loss over time
      - Actor loss over time
      - Episodic reward over time

    Args:
        client_instance: Instance of Client with current state and logs
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax_pos = axs[0, 0]
    ax_critic = axs[0, 1]
    ax_actor = axs[1, 0]
    ax_reward = axs[1, 1]

    # Configure position subplot
    ax_pos.set_xlim(FIELD_X_LIMITS[0] - 0.2, FIELD_X_LIMITS[1] + 0.2)
    ax_pos.set_ylim(FIELD_Y_LIMITS[0] - 0.2, FIELD_Y_LIMITS[1] + 0.2)
    ax_pos.set_title("Live Position")
    ax_pos.set_xlabel("x [m]")
    ax_pos.set_ylabel("y [m]")
    ax_pos.grid(True)

    # Draw field boundary
    border_x = [FIELD_X_LIMITS[0], FIELD_X_LIMITS[1], FIELD_X_LIMITS[1], FIELD_X_LIMITS[0], FIELD_X_LIMITS[0]]
    border_y = [FIELD_Y_LIMITS[0], FIELD_Y_LIMITS[0], FIELD_Y_LIMITS[1], FIELD_Y_LIMITS[1], FIELD_Y_LIMITS[0]]
    ax_pos.plot(border_x, border_y, '--', linewidth=1.2, label='Field Boundary')
    goal_dot, = ax_pos.plot([], [], 'ro', label='Goal')
    car_dot, = ax_pos.plot([], [], 'bo', label='Agent')
    ax_pos.legend()

    # Configure loss and reward subplots
    ax_critic.set_title("Critic Loss")
    ax_critic.set_xlabel("Time [s]")
    ax_critic.grid(True)
    critic_line, = ax_critic.plot([], [], label='Critic Loss')
    ax_critic.legend()

    ax_actor.set_title("Actor Loss")
    ax_actor.set_xlabel("Time [s]")
    ax_actor.grid(True)
    actor_line, = ax_actor.plot([], [], label='Actor Loss')
    ax_actor.legend()

    ax_reward.set_title("Episodic Reward")
    ax_reward.set_xlabel("Time [s]")
    ax_reward.grid(True)
    reward_line, = ax_reward.plot([], [], label='Reward')
    ax_reward.legend()

    def update(frame):
        # Update agent and goal positions
        pos = client_instance.current_position
        goal = client_instance.current_goal
        if pos is not None and goal is not None:
            car_dot.set_data(pos[0], pos[1])
            goal_dot.set_data(goal[0], goal[1])

        # Update metrics
        if client_instance.loss_log:
            base_time = client_instance.loss_log[0]['timestamp']
            times = [entry['timestamp'] - base_time for entry in client_instance.loss_log]
            critic_vals = [entry['critic_loss'] for entry in client_instance.loss_log]
            actor_vals = [entry['actor_loss'] for entry in client_instance.loss_log]
            reward_vals = [entry['reward'] for entry in client_instance.loss_log]
            critic_line.set_data(times, critic_vals)
            actor_line.set_data(times, actor_vals)
            reward_line.set_data(times, reward_vals)

            for ax in (ax_critic, ax_actor, ax_reward):
                ax.relim()
                ax.autoscale_view()

        return car_dot, goal_dot, critic_line, actor_line, reward_line

    ani = animation.FuncAnimation(fig, update, interval=500)
    plt.tight_layout()
    plt.show()


# === Main Execution ===
if __name__ == "__main__":
    # Instantiate client
    client = Client()
    # Start send_loop in background thread
    thread = threading.Thread(target=client.send_loop, daemon=True)
    thread.start()
    # Launch live visualization
    start_live_plot(client)