import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import triad_openvr
import time
import sys
import json
import os
import socket
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading

# === Hyperparameter ===
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = 100000
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Netzwerke ===
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(a), dtype=torch.float32).to(DEVICE),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.tensor(np.array(s2), dtype=torch.float32).to(DEVICE),
            torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        )

    def __len__(self):
        return len(self.buffer)

class Client:
    def __init__(self, host="192.168.137.5", port=12345):
        self.vr = triad_openvr.triad_openvr()
        self.vr.print_discovered_objects()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

        self.interval = 1 / 50
        self.goal = self._sample_goal()
        self.goal_start_time = time.time()
        self.return_goal = np.array([-0.3, -0.3])
        self.goal_hold_start = None
        self.goal_hold_duration = 5.0

        self.paused = False
        self.reset_penalty = 0.0
        self.state = "orienting"
        self.prev_speed_sign = 1
        self.in_neutral_phase = False
        self.neutral_count = 0

        self.prev_theta = 0
        self.prev_position = np.zeros(2)
        self.prev_dist = None
        self.prev_v_sign = 1
        self.v_switches = 0
        self.last_steering = None
        self.same_steering_count = 0
        self.last_speed = None
        self.same_speed_count = 0

        self.actor = Actor(7, 2).to(DEVICE)
        self.actor_target = Actor(7, 2).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(7, 2).to(DEVICE)
        self.critic_target = Critic(7, 2).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.prev_obs = None

        self.current_position = np.array([0.0, 0.0])
        self.current_goal = self.goal

    @staticmethod
    def _sample_goal():
        while True:
            g = np.random.uniform(-1.0, 1.0, size=2)
            if np.linalg.norm(g) > 1.0:
                return g

    @staticmethod
    def angle_difference(a, b):
        return (a - b + 180) % 360 - 180

    def _compute_reward(self, position, theta, v):
        dist = np.linalg.norm(self.goal - position)
        reward = 0.0

        if self.prev_dist is not None:
            delta = self.prev_dist - dist
            if delta > 0:
                reward += delta * 15.0  # sehr starke Belohnung für Annäherung
            else:
                reward -= delta * 500.0  # dreifach bestraft fürs Entfernen
        self.prev_dist = dist

        reward += (1.5 - dist) * 2.0

        if abs(v) < 0.01 and dist > 0.2:
            reward -= 1.5
        if v < -0.1:
            reward -= 1.0
        elif v > 0.1:
            reward += 1.0
        if dist < 0.3 and abs(v) < 0.7:
            reward += 50.0
        elif dist < 0.3 and abs(v) < 0.9:
            reward += 10.0

        sign = np.sign(v)
        if sign != self.prev_v_sign and abs(v) > 0.1:
            self.v_switches += 1
            reward -= 5.0
        self.prev_v_sign = sign
        if self.v_switches >= 5:
            reward -= 10.0
            self.v_switches = 0

        angle_diff = abs(theta - self.prev_theta)
        move_diff = np.linalg.norm(position - self.prev_position)
        if angle_diff > 0.3 and move_diff < 0.05:
            reward -= 3.0

        if self.paused:
            reward -= 2000.0

        if self.last_steering is not None and abs(self.last_steering - self.current_steering) < 0.3:
            self.same_steering_count += 1
            if self.same_steering_count > 10:
                reward -= 50.0
        else:
            self.same_steering_count = 0
        self.last_steering = self.current_steering

        if self.last_speed is not None and abs(self.last_speed - v) < 0.15:
            self.same_speed_count += 1
            if self.same_speed_count > 3:
                reward -= 20.0
        else:
            self.same_speed_count = 0
        self.last_speed = v

        self.prev_theta = theta
        self.prev_position = position.copy()

        return reward

    def send_speed_message(self, speed, steering):
        current_sign = np.sign(speed)

        if current_sign != 0 and current_sign != self.prev_speed_sign:
            print("\n⏹️ Richtungswechsel erkannt – sende 0-Signale.")
            self.in_neutral_phase = True
            self.neutral_count = 0
            self.prev_speed_sign = current_sign

        if self.in_neutral_phase:
            if self.neutral_count < 15:
                neutral = json.dumps({"speed": 0.0, "steering": 0.0}) + "\n"
                self.sock.sendall(neutral.encode())
                self.neutral_count += 1
                time.sleep(0.02)
                return
            else:
                self.in_neutral_phase = False

        self.prev_speed_sign = current_sign
        message = json.dumps({"speed": float(speed), "steering": float(steering)}) + "\n"
        self.sock.sendall(message.encode())

    # Hier Reward-Anzeige ergänzen (Beispielhaft in send_loop):


    def send_loop(self):
        try:
            while True:
                start = time.time()
                pose = self.vr.devices["tracker_1"].get_pose_euler()
                x, y, yaw = pose[0], pose[1], pose[3]
                position = np.array([x, y])
                self.current_position = position
                self.current_goal = self.goal

                if not (-1.5 <= x <= 1.5 and -1.5 <= y <= 1.5):
                    if not self.paused:
                        print("\n⛔ Spielfeld verlassen – Rückführung aktiviert.")
                    self.paused = True
                    self.state = "orienting"
                    self.goal_hold_start = None
                elif not self.paused and time.time() - self.goal_start_time > 60:
                    print("\n⏱️ Zeitlimit überschritten – Rückführung aktiviert.")
                    self.paused = True
                    self.state = "orienting"
                    self.goal_hold_start = None

                if self.paused:
                    dx = self.return_goal[0] - x
                    dy = self.return_goal[1] - y
                    distance = np.linalg.norm([dx, dy])
                    car_angle = (yaw - 150)
                    car_angle = -car_angle % 360
                    goal_angle = np.degrees(np.arctan2(dy, dx)) % 360
                    angle_diff = (goal_angle - car_angle + 540) % 360 - 180
                    orientation_threshold = 3

                    if distance < 0.45:
                        speed = steering = 0.0
                        self.paused = False
                        self.prev_obs = None
                        self.goal_start_time = time.time()
                    else:
                        if self.state == "orienting":
                            if abs(angle_diff) < orientation_threshold:
                                speed = 0.85
                                steering = 0.0
                                self.state = "moving"
                            else:
                                speed = 0.85
                                damb = angle_diff / 180.0
                                steering = -damb * 1.8
                        else:
                            speed = 0.85
                            steering = 0.0
                else:
                    dx, dy = self.goal[0] - x, self.goal[1] - y
                    dist = np.linalg.norm([dx, dy])
                    car_angle = (-yaw - 150) % 360
                    goal_angle = np.degrees(np.arctan2(dy, dx)) % 360
                    angle_diff = self.angle_difference(goal_angle, car_angle)

                    state = np.array([self.goal[0], self.goal[1], x, y, yaw, dist, angle_diff], dtype=np.float32)
                    obs_tensor = torch.tensor(state).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        action = self.actor(obs_tensor).cpu().numpy()[0]
                    action += np.random.normal(0, 0.1, size=2)
                    action = np.clip(action, -1.0, 1.0)
                    speed, steering = action
                    self.current_steering = steering

                    reward = self._compute_reward(position, yaw, speed)
                    next_obs = state.copy()
                    done = False

                    if self.prev_obs is not None:
                        self.buffer.push((self.prev_obs, action, reward, next_obs, float(done)))
                    self.prev_obs = next_obs

                    if len(self.buffer) >= BATCH_SIZE:
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

                        for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
                            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
                        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

                    if dist < 0.3:
                        if self.goal_hold_start is None:
                            self.goal_hold_start = time.time()
                        elif time.time() - self.goal_hold_start >= self.goal_hold_duration:
                            # ✅ Belohnung bei Zielerreichung einmalig hinzufügen
                            if self.prev_obs is not None:
                                success_reward = 1500.0  # hoher Reward
                                done = True
                                self.buffer.push((self.prev_obs, action, success_reward, next_obs, float(done)))

                            self.goal = self._sample_goal()
                            self.goal_start_time = time.time()
                            self.goal_hold_start = None

                self.send_speed_message(speed, steering)
                print(f"\rPos: ({x:.2f}, {y:.2f}) | Ziel: ({self.goal[0]:.2f}, {self.goal[1]:.2f}) | Speed: {speed:.2f}, Steering: {steering:.2f}| R: ({reward:.2f})", end="")
                time.sleep(max(0, self.interval - (time.time() - start)))

        except KeyboardInterrupt:
            print("\n🛑 Stoppe...")
            self.sock.close()

def start_live_plot(client_instance):
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    goal_dot, = ax.plot([], [], 'ro', label='Ziel')
    car_dot, = ax.plot([], [], 'bo', label='Auto')
    ax.legend()
    ax.grid(True)
    plt.title("Live Position vs. Zielpunkt")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    def update_plot(frame):
        pos = client_instance.current_position
        goal = client_instance.current_goal
        if pos is not None and goal is not None:
            car_dot.set_data([float(pos[0])], [float(pos[1])])
            goal_dot.set_data([float(goal[0])], [float(goal[1])])
        return car_dot, goal_dot

    ani = animation.FuncAnimation(fig, update_plot, interval=100, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    client = Client()
    send_thread = threading.Thread(target=client.send_loop, daemon=True)
    send_thread.start()
    start_live_plot(client)
