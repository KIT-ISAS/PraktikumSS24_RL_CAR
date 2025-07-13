# === Verbesserter DDPG RL-Agent mit Rückführung und optimiertem Verhalten ===

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import triad_openvr
import time
import json
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
TAU = 0.01
BUFFER_SIZE = 800_000
BATCH_SIZE = 128
SAVE_INTERVAL = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Modelle ===
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

def save_models(actor, critic, episode):
    torch.save(actor.state_dict(), f"actor_episode_{episode}.pth")
    torch.save(critic.state_dict(), f"critic_episode_{episode}.pth")
    print(f"\n💾 Modelle gespeichert (Episode {episode})")

def compute_reward(position, goal, prev_dist, speed, steering):
    dist = np.linalg.norm(goal - position)
    reward = 0.0

    # Strafe fürs Stehenbleiben fern vom Ziel
    if dist > 0.3 and abs(speed) < 0.7:
        reward -= 5.0  # Deutliche Strafe fürs Nichtstun

    if prev_dist is not None:
        progress = prev_dist - dist
        reward += progress * 10.0
        if progress < 0:
            reward -= 2.0

   # reward += max(0, (1.5 - dist)) * 10.0

    if dist < 0.3 and abs(speed) < 0.1 and abs(steering) < 0.1:
        reward += 10.0

    if speed < -0.1:
        reward -= 2.0
    if abs(speed) < 0.05 and dist > 0.5:
        reward -= 1.0
    if abs(steering) > 0.6:
        reward -= abs(steering) * 2.0

    return reward, dist
# === Client-Klasse ===
class Client:
    def __init__(self, host="192.168.137.5", port=12345):
        self.vr = triad_openvr.triad_openvr()
        self.vr.print_discovered_objects()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

        self.actor = Actor(7, 2).to(DEVICE)
        self.critic = Critic(7, 2).to(DEVICE)
        self.actor_target = Actor(7, 2).to(DEVICE)
        self.critic_target = Critic(7, 2).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.episode = 0
        self.prev_obs = None
        self.prev_dist = None
        self.goal = self._sample_goal()
        self.goal_start_time = time.time()

        self.prev_speed_sign = 1
        self.in_neutral_phase = False
        self.neutral_count = 0
        self.current_position = np.array([0.0, 0.0])
        self.current_goal = self.goal

        self.paused = False
        self.state = "orienting"
        self.return_goal = np.array([-0.3, -0.3])
        self.exploration_noise = 0.2

    def _sample_goal(self):
        while True:
            g = np.random.uniform(-1.0, 1.0, size=2)
            if np.linalg.norm(g) > 0.5:
                return g

    def send_speed_message(self, speed, steering):
        current_sign = np.sign(speed)
        if current_sign != 0 and current_sign != self.prev_speed_sign:
            print("\n⏹️ Richtungswechsel erkannt – sende 0-Signale.")
            self.in_neutral_phase = True
            self.neutral_count = 0
            self.prev_speed_sign = current_sign
        if self.in_neutral_phase:
            if self.neutral_count < 4:
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
        try:
            self.goal_start_time = time.time()
            self.goal_reached_time = None
            self.hold_timer_active = False

            while True:
                pose = self.vr.devices["tracker_2"].get_pose_euler()
                x, y, yaw = pose[0], pose[1], pose[3]
                position = np.array([x, y])
                self.current_position = position
                self.current_goal = self.goal

                if not (-1.5 <= x <= 1.5 and -1.5 <= y <= 1.5):
                    print("\n⛔ Spielfeld verlassen – Rueckfuehrung aktiviert.")
                    self.paused = True
                    self.state = "orienting"
                    self.goal_reached_time = None
                    self.hold_timer_active = False

                if self.paused:
                    dx = self.return_goal[0] - x
                    dy = self.return_goal[1] - y
                    distance = np.linalg.norm([dx, dy])
                    car_angle = (yaw - 150)
                    car_angle = -car_angle % 360
                    goal_angle = np.degrees(np.arctan2(dy, dx)) % 360
                    angle_diff = (goal_angle - car_angle + 540) % 360 - 180
                    orientation_threshold = 3
                    time.sleep(0.02)

                    if distance < 0.45:
                        speed = steering = 0.0
                        self.paused = False
                        self.prev_obs = None
                        self.goal_start_time = time.time()
                        continue
                    else:
                        if self.state == "orienting":
                            if abs(angle_diff) < orientation_threshold:
                                speed = 0.95
                                steering = 0.0
                            else:
                                speed = 0.85
                                damb = angle_diff / 180.0
                                steering = -damb * 3

                    self.send_speed_message(speed, steering)
                    print(f"\r[RUECKFUEHRUNG] Pos: ({x:.2f},{y:.2f}) → Rueckkehr", end="")
                    continue

                dx, dy = self.goal[0] - x, self.goal[1] - y
                dist = np.linalg.norm([dx, dy])
                car_angle = (-yaw - 150) % 360
                goal_angle = np.degrees(np.arctan2(dy, dx)) % 360
                angle_diff = (goal_angle - car_angle + 180) % 360 - 180

                # Normalisierung
                yaw_norm = yaw / 180.0
                angle_diff_norm = angle_diff / 180.0
                state = np.array([self.goal[0], self.goal[1], x, y, yaw_norm, dist, angle_diff_norm], dtype=np.float32)
                obs_tensor = torch.tensor(state).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    noise = max(0.05, self.exploration_noise * np.exp(-self.episode / 200))
                    if len(self.buffer) < BATCH_SIZE:
                        action = np.random.uniform(-1, 1, size=2)
                    else:
                        action = self.actor(obs_tensor).cpu().numpy()[0]
                        action = np.clip(action + np.random.normal(0, noise, size=2), -1.0, 1.0)

                if len(self.buffer) >= BATCH_SIZE:
                    self.train_step()

                speed, steering = action
                reward, current_dist = compute_reward(position, self.goal, self.prev_dist, speed, steering)
                self.prev_dist = current_dist
                next_state = state.copy()
                done = current_dist < 0.2 or (time.time() - self.goal_start_time) > 45

                if self.prev_obs is not None:
                    self.buffer.push((self.prev_obs, action, reward, next_state, float(done)))
                self.prev_obs = next_state

                if (time.time() - self.goal_start_time) > 45:
                    print("\n⏱️ Zeitlimit ueberschritten – Rueckfuehrung aktiviert.")
                    self.paused = True
                    self.state = "orienting"
                    self.goal_reached_time = None
                    self.hold_timer_active = False
                    continue

                at_goal = current_dist < 0.2
                almost_still = abs(speed) < 1

                if at_goal and almost_still:
                    if not self.hold_timer_active:
                        self.goal_reached_time = time.time()
                        self.hold_timer_active = True
                    elif time.time() - self.goal_reached_time >= 0.5:
                        print("\n✅ Ziel erfolgreich erreicht und gehalten – neues Ziel.")
                        self.goal = self._sample_goal()
                        self.prev_obs = None
                        self.prev_dist = None
                        self.goal_start_time = time.time()
                        self.goal_reached_time = None
                        self.hold_timer_active = False
                        self.episode += 1
                        save_models(self.actor, self.critic, self.episode)
                else:
                    self.goal_reached_time = None
                    self.hold_timer_active = False

                if len(self.buffer) >= 100:
                    self.train_step()

                self.send_speed_message(speed, steering)
                print(
                    f"\rEp: {self.episode} | Pos: ({x:.2f},{y:.2f}) | Goal: {self.goal} | R: {reward:.2f} | a: [{speed:.2f}, {steering:.2f}]",
                    end="")
                time.sleep(0.02)

        except KeyboardInterrupt:
            print("\n🛑 Stoppe...")
            self.sock.close()

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return  # Sicherheitsabbruch

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


# === Visualisierung ===
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

# === Main ===
if __name__ == "__main__":
    client = Client()
    thread = threading.Thread(target=client.send_loop, daemon=True)
    thread.start()
    start_live_plot(client)
