# === DDPG Reinforcement Learning Agent für Autonomes Fahrzeug mit wissenschaftlichem Logging und Rückführung ===

# === Imports ===
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
import os
from datetime import datetime
import pickle

# === Hyperparameter ===
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 256
TRAIN_INTERVAL = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Spielfeldgrenzen ===
FIELD_X_LIMITS = (-1.2, 1.2)
FIELD_Y_LIMITS = (-1.5, 1.5)

# === Totzone für Geschwindigkeit ===
SPEED_THRESHOLD = 0.85

# === Neutrale Startwerte für globale Reward-Logik ===
no_move_start_time = None
prev_speed_sign = 0
steering_hold_counter = 0
prev_steering = 0.0
v_switch_counter = 0

# === Actor-Netzwerk ===
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

# === Critic-Netzwerk ===
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

# === Replay Buffer ===
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

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)
        print(f"💾 ReplayBuffer gespeichert unter {filepath}")

    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.buffer = pickle.load(f)
            print(f"📂 ReplayBuffer geladen von {filepath}")

    def __len__(self):
        return len(self.buffer)

# === Reward-Funktion ===
def compute_reward(position, goal, prev_dist, speed, steering, yaw, holding=False):
    global no_move_start_time, steering_hold_counter, prev_steering
    global prev_speed_sign, v_switch_counter

    reward = 0.0
    dist = np.linalg.norm(goal - position)

    # === 1. Ziel erreicht und gehalten ===
    if holding:
        return 20.0, dist

    # === 2. Annäherung ans Ziel ===
    reward += 3.0 / max(dist, 0.1)

    # === 3. Spielfeldbegrenzung ===
    if not (FIELD_X_LIMITS[0] <= position[0] <= FIELD_X_LIMITS[1] and
            FIELD_Y_LIMITS[0] <= position[1] <= FIELD_Y_LIMITS[1]):
        return -20.0, dist

    # === 4. Fortschritt zur Zielposition ===
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

    # === 5. Langsames oder kein Fortschritt ===
    if no_move_start_time is not None:
        still_time = time.time() - no_move_start_time
        if still_time > 2:
            reward -= min((still_time - 20) * 0.1, 20.0)

    # === 6. Totzone bestrafen ===
    if dist > 0.3 and abs(speed) < 0.55:
        reward -= 5.0

    # === 8. Richtungswechsel kontrollieren ===
    current_sign = np.sign(speed)
    if current_sign != 0 and current_sign != prev_speed_sign:
        v_switch_counter += 1
        reward -= min(2.0 * v_switch_counter, 5.0)
        prev_speed_sign = current_sign
    else:
        v_switch_counter = max(0, v_switch_counter - 1)

    # === 9. Circling erkennen durch konstantes Lenken ===
    if abs(steering - prev_steering) < 0.2:
        steering_hold_counter += 1
    else:
        steering_hold_counter = 0

    if steering_hold_counter > 30:
        reward -= 2.0 + (steering_hold_counter - 30) * 0.1

    prev_steering = steering

    # === 10. Reward Clipping ===
    reward = np.clip(reward, -20.0, 20.0)
    return reward, dist

    # === 11. Ausrichtung zum Zielpunkt belohnen ===
    to_goal = goal - position
    to_goal_angle = np.arctan2(to_goal[1], to_goal[0])

    # Winkel zwischen Fahrtrichtung (yaw) und Richtung zum Ziel
    angle_diff = np.arctan2(np.sin(to_goal_angle - yaw), np.cos(to_goal_angle - yaw))  # [-pi, pi]
    angle_diff_deg = np.degrees(abs(angle_diff))  # Optional: in Grad, zur besseren Skalierung

    # Je kleiner der Winkel, desto höher der Bonus (max +3, min 0)
    orientation_reward = max(0.0, 3.0 * (1 - angle_diff_deg / 180))
    reward += orientation_reward

def reset_reward_tracking():
    global steering_hold_counter, prev_steering, v_switch_counter, prev_speed_sign
    steering_hold_counter = 0
    prev_steering = 0.0
    v_switch_counter = 0
    prev_speed_sign = 0


# === Platzhalter für Client-Klasse ===
class Client:
    def __init__(self, host="192.168.137.5", port=12345):
        # === Verbindung & Sensoren ===
        self.vr = triad_openvr.triad_openvr()
        self.vr.print_discovered_objects()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

        # === Netzwerke ===
        self.actor = Actor(5, 2).to(DEVICE)
        self.critic = Critic(5, 2).to(DEVICE)
        self.actor_target = Actor(5, 2).to(DEVICE)
        self.critic_target = Critic(5, 2).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # === Erfahrungsspeicher ===
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.replay_path = "models/replay_buffer.pkl"

        # === Trainingsstatus ===
        self.episode = 0
        self.iteration = 0
        self.loss_log = []
        self.prev_reward = 0.0

        # === Zustände und Navigation ===
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

        # === Bewegung / Richtungswechsel ===
        self.prev_speed_sign = 0
        self.in_neutral_phase = False
        self.neutral_count = 0

        # === Logging & Verzeichnisstruktur ===
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_log_filename = f"plot_results/measurement_data/episode_{self.episode}_{timestamp_str}.json"
        os.makedirs(os.path.dirname(self.data_log_filename), exist_ok=True)
        self.raw_data_log = []

    def _sample_goal(self):
        while True:
            g = np.random.uniform(-1.0, 1.0, size=2)
            if np.linalg.norm(g) > 0.5:
                return g

    def load_models(self):
        actor_path = "models/actor_latest.pt"
        critic_path = "models/critic_latest.pt"
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_state_dict(torch.load(actor_path))
            self.critic.load_state_dict(torch.load(critic_path))
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            print("📥 Modelle geladen (fortlaufendes Training).")

    def save_models(self):
        os.makedirs("models", exist_ok=True)
        torch.save(self.actor.state_dict(), f"models/actor_ep{self.episode}.pt")
        torch.save(self.critic.state_dict(), f"models/critic_ep{self.episode}.pt")
        torch.save(self.actor.state_dict(), "models/actor_latest.pt")
        torch.save(self.critic.state_dict(), "models/critic_latest.pt")
        print("💾 Modelle gespeichert.")

    def save_logs(self):
        os.makedirs("plot_results/loss_logs", exist_ok=True)
        with open(f"plot_results/loss_logs/loss_log_ep{self.episode}.json", "w") as f:
            json.dump(self.loss_log, f, indent=2)
            print("📝 Loss-Log gespeichert.")

    def send_speed_message(self, speed, steering):
        current_sign = np.sign(speed)
        if current_sign != 0 and current_sign != self.prev_speed_sign:
            print("\n⏹️ Richtungswechsel erkannt – sende 0-Signale.")
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
        try:
            self.goal_start_time = time.time()
            self.timer_no_move = time.time()
            self.goal_reached_time = None
            self.hold_timer_active = False
            prev_x = prev_y = prev_time = None

            while True:
                pose = self.vr.devices["tracker_1"].get_pose_euler()
                x, y, yaw = pose[0], pose[1], pose[3]
                timestamp = time.time()
                position = np.array([x, y])
                self.current_position = position
                self.current_goal = self.goal

                # Geschwindigkeit berechnen
                if prev_x is not None:
                    dx = x - prev_x
                    dy = y - prev_y
                    dt = timestamp - prev_time
                    speed_calculated = np.sqrt(dx ** 2 + dy ** 2) / dt if dt > 0 else 0.0
                else:
                    speed_calculated = 0.0
                prev_x, prev_y, prev_time = x, y, timestamp

                # Nur bestrafen, wenn wir *noch nicht* in der Rückführung sind
                # Nur einmal bestrafen, wenn noch nicht gehandelt
                if not self.paused and not self.out_of_bounds_handled and not (
                        FIELD_X_LIMITS[0] <= x <= FIELD_X_LIMITS[1] and FIELD_Y_LIMITS[0] <= y <= FIELD_Y_LIMITS[1]
                ):
                    print("\n⛔ Spielfeld verlassen – Bestrafung + Rückführung aktiviert.")
                    reward, _ = compute_reward(position, self.goal, self.prev_dist, speed_calculated, steering, yaw, holding=False)
                    done = True
                    next_state = self.prev_obs if self.prev_obs is not None else state

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
                    self.data_log_filename = f"plot_results/measurement_data/episode_{self.episode}_{timestamp_str}.json"

                    self.prev_obs = None
                    self.prev_dist = None
                    self.goal_reached_time = None
                    self.hold_timer_active = False

                    self.save_models()

                    self.paused = True
                    self.state = "orienting"
                    self.out_of_bounds_handled = True  # Verhindert Wiederholung
                    continue

                # Rückführlogik
                if self.paused:
                    reset_reward_tracking()
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
                        self.out_of_bounds_handled = False  # Flag zurücksetzen für nächste Runde
                        continue
                    else:
                        if self.state == "orienting":
                            if abs(angle_diff) < orientation_threshold:
                                speed = 0.95
                                steering = 0.0
                            else:
                                speed = 1
                                damb = angle_diff / 180.0
                                steering = -damb * 3

                    self.send_speed_message(speed, steering)
                    print(f"\r[RUECKFUEHRUNG] Pos: ({x:.2f},{y:.2f}) → Rueckkehr", end="")
                    continue

                # RL-Zustand
                dx, dy = self.goal[0] - x, self.goal[1] - y
                dist = np.linalg.norm([dx, dy])
                car_angle = (-yaw - 150) % 360
                goal_angle = np.degrees(np.arctan2(dy, dx)) % 360
                angle_diff = (goal_angle - car_angle + 180) % 360 - 180

                yaw_norm = yaw / 180.0
                angle_diff_norm = angle_diff / 180.0
                state = np.array([self.goal[0], self.goal[1], x, y, yaw_norm], dtype=np.float32)
                obs_tensor = torch.tensor(state).unsqueeze(0).to(DEVICE)

                # Aktion wählen
                if len(self.buffer) < BATCH_SIZE:
                    action = np.random.uniform(-1, 1, size=2)
                else:
                    with torch.no_grad():
                        action = self.actor(obs_tensor).cpu().numpy()[0]
                        noise = max(0.05, 0.2 * np.exp(-self.episode / 300))
                        action += np.random.normal(0, noise, size=2)
                        action = np.clip(action, -1.0, 1.0)

                speed, steering = action
                reward, current_dist = compute_reward(position, self.goal, self.prev_dist, speed, steering, yaw)
                self.prev_dist = current_dist
                next_state = state.copy()
                done = current_dist < 0.2

                if self.prev_obs is not None:
                    self.buffer.push((self.prev_obs, action, reward, next_state, float(done)))

                self.prev_obs = next_state
                self.prev_reward = reward

                # Logging
                self.raw_data_log.append({
                    "timestamp": timestamp,
                    "episode": self.episode,
                    "position": {"x": x, "y": y},
                    "yaw": yaw,
                    "goal": {"x": float(self.goal[0]), "y": float(self.goal[1])},
                    "speed_calculated": speed_calculated,
                    "paused": self.paused,
                    "action": {"speed": float(speed), "steering": float(steering)},
                    "reward": float(reward),
                    "critic_loss": float(self.loss_log[-1]["critic_loss"]) if self.loss_log else None,
                    "actor_loss": float(self.loss_log[-1]["actor_loss"]) if self.loss_log else None
                })

                # Training
                if self.iteration % TRAIN_INTERVAL == 0 and len(self.buffer) >= BATCH_SIZE:
                    self.train_step()

                self.send_speed_message(speed, steering)
                print(f"\r[STATUS] Pos: ({position[0]:.2f}, {position[1]:.2f}) | Ziel: ({self.goal[0]:.2f}, {self.goal[1]:.2f}) | Distanz: {current_dist:.3f} | Done: {done}", end="", flush=True)

                # Ziel erreicht
                if done:
                    if self.goal_reached_time is None:
                        self.goal_reached_time = time.time()
                        self.hold_timer_active = True
                    elif time.time() - self.goal_reached_time >= 0.5:
                        print("\n✅ Ziel erreicht und gehalten.")
                        reward = 10
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
                        self.data_log_filename = f"plot_results/measurement_data/episode_{self.episode}_{timestamp_str}.json"
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
                        reward, _ = compute_reward(position, self.goal, self.prev_dist, 0.0, 0.0, yaw, holding=True)
                        self.buffer.push((self.prev_obs, [0.0, 0.0], reward, next_state, 0.0))
                        time.sleep(0.005)
                        continue
                else:
                    self.goal_reached_time = None
                    self.hold_timer_active = False

                reset_reward_tracking()
                time.sleep(0.005)

        except KeyboardInterrupt:
            print("\n🛑 Stoppe...")
            self.sock.close()
            self.save_logs()
            self.save_raw_data()

    def train_step(self):
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

        self.loss_log.append({
            "timestamp": time.time(),
            "episode": self.episode,
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "reward": self.prev_reward
        })

    def save_raw_data(self):
        with open(self.data_log_filename, "w") as f:
            json.dump(self.raw_data_log, f, indent=2)
            self.buffer.save(self.replay_path)
            print(f"📁 Bewegungsdaten gespeichert unter: {self.data_log_filename}")
# === Visualisierung ===
def start_live_plot(client_instance):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax_pos = axs[0, 0]
    ax_critic = axs[0, 1]
    ax_actor = axs[1, 0]
    ax_reward = axs[1, 1]

    # Spielfeldgrenzen setzen
    ax_pos.set_xlim(FIELD_X_LIMITS[0] - 0.2, FIELD_X_LIMITS[1] + 0.2)
    ax_pos.set_ylim(FIELD_Y_LIMITS[0] - 0.2, FIELD_Y_LIMITS[1] + 0.2)
    ax_pos.set_title("Live Position")
    ax_pos.set_xlabel("x [m]")
    ax_pos.set_ylabel("y [m]")
    ax_pos.grid(True)

    # Spielfeldgrenze einzeichnen
    border_x = [FIELD_X_LIMITS[0], FIELD_X_LIMITS[1], FIELD_X_LIMITS[1], FIELD_X_LIMITS[0], FIELD_X_LIMITS[0]]
    border_y = [FIELD_Y_LIMITS[0], FIELD_Y_LIMITS[0], FIELD_Y_LIMITS[1], FIELD_Y_LIMITS[1], FIELD_Y_LIMITS[0]]
    ax_pos.plot(border_x, border_y, 'k--', linewidth=1.2, label='Spielfeldgrenze')

    goal_dot, = ax_pos.plot([], [], 'ro', label='Ziel')
    car_dot, = ax_pos.plot([], [], 'bo', label='Auto')
    ax_pos.legend()

    critic_line, = ax_critic.plot([], [], label='Critic Loss', color='blue')
    actor_line, = ax_actor.plot([], [], label='Actor Loss', color='green')
    reward_line, = ax_reward.plot([], [], label='Reward', color='red')

    for ax in [ax_critic, ax_actor, ax_reward]:
        ax.set_xlabel("Episode")
        ax.grid(True)
        ax.legend()

    ax_critic.set_title("Critic Loss")
    ax_actor.set_title("Actor Loss")
    ax_reward.set_title("Episodischer Reward")

    def update(frame):
        pos = client_instance.current_position
        goal = client_instance.current_goal

        if pos is not None and goal is not None:
            car_dot.set_data([float(pos[0])], [float(pos[1])])
            goal_dot.set_data([float(goal[0])], [float(goal[1])])

        timestamps = [entry["timestamp"] - client_instance.loss_log[0]["timestamp"] for entry in client_instance.loss_log]
        critic_losses = [entry["critic_loss"] for entry in client_instance.loss_log]
        actor_losses = [entry["actor_loss"] for entry in client_instance.loss_log]
        rewards = [entry["reward"] for entry in client_instance.loss_log]

        critic_line.set_data(timestamps, critic_losses)
        actor_line.set_data(timestamps, actor_losses)
        reward_line.set_data(timestamps, rewards)

        for ax in [ax_critic, ax_actor, ax_reward]:
            ax.relim()
            ax.autoscale_view()

        return car_dot, goal_dot, critic_line, actor_line, reward_line

    ani = animation.FuncAnimation(fig, update, interval=500, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

# === Main ===
if __name__ == "__main__":
    client = Client()
    thread = threading.Thread(target=client.send_loop, daemon=True)
    thread.start()
    start_live_plot(client)

