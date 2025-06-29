import torch
import numpy as np
import triad_openvr
import time
import sys
import json
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import socket
from torch import nn


# === Modellstruktur ===
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


# === Gerät und Modell laden ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor = Actor(state_dim=6, action_dim=2).to(DEVICE)

model_path = "actor_100000.pth"
if os.path.exists(model_path):
    actor.load_state_dict(torch.load(model_path, map_location=DEVICE))
    actor.eval()
    print(f"✅ Modell geladen: {model_path}")
else:
    print(f"⚠️ Kein Modell gefunden: {model_path} – Zufällige Aktionen werden verwendet.")


# === Zielposition generieren ===
def sample_goal(area_size=4.0):
    while True:
        goal = np.random.uniform(-area_size / 2 + 0.5, area_size / 2 - 0.5, size=2)
        if np.linalg.norm(goal) > 1.0:
            return goal


# === Reward-Funktion ===
class RewardEvaluator:
    def __init__(self, goal):
        self.goal = goal
        self.prev_dist = None
        self.prev_theta = 0.0
        self.prev_position = np.array([0.0, 0.0])
        self.prev_v_sign = 1
        self.v_switches = 0

    def compute_reward(self, position, theta, v):
        dist_to_goal = np.linalg.norm(self.goal - position)
        reward = -dist_to_goal

        if self.prev_dist is not None:
            delta = self.prev_dist - dist_to_goal
            if delta > 0.05:
                reward += delta * 10.0
        self.prev_dist = dist_to_goal

        if abs(v) < 0.01 and dist_to_goal > 0.2:
            reward -= 1.5

        # ⛔ Bestrafung für Rückwärtsfahrt
        if v < -0.1:
            reward -= 1.0
        # ✅ Bonus für Vorwärtsfahrt
        elif v > 0.1:
            reward += 1.0

        # ✅ NEU: Zielgeschwindigkeit ("Goldilocks-Zone")
        if 0.3 < v < 1.0:
            reward += 20.0  # Belohnung für ideale Geschwindigkeit

        # 🎯 Belohnung bei Zielerreichung mit geringer Geschwindigkeit
        if dist_to_goal < 0.2 and abs(v) < 0.1:
            reward += 5.0

        current_v_sign = np.sign(v)
        if current_v_sign != self.prev_v_sign and abs(v) > 0.1:
            self.v_switches += 1
            reward -= 5.0
        self.prev_v_sign = current_v_sign

        if self.v_switches >= 5:
            reward -= 10.0
            self.v_switches = 0

        angle_diff = abs(theta - self.prev_theta)
        move_diff = np.linalg.norm(position - self.prev_position)
        if angle_diff > 0.3 and move_diff < 0.05:
            reward -= 3.0

        self.prev_theta = theta
        self.prev_position = position.copy()

        return reward


# === Hauptklasse mit Socket-Kommunikation ===
class Client:
    def __init__(self, raspberry_host="192.168.137.5", raspberry_port=12345):
        self.vr = triad_openvr.triad_openvr()
        self.vr.print_discovered_objects()
        self.interval = 1 / 250

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((raspberry_host, raspberry_port))
            print(f"✅ Verbunden mit Raspberry Pi ({raspberry_host}:{raspberry_port})")
        except Exception as e:
            print(f"❌ Verbindung fehlgeschlagen: {e}")
            sys.exit(1)

        self.x_data, self.y_data = [], []
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100)

        self.area_size = 4.0
        self.goal = sample_goal(self.area_size)
        self.goal_hold_timer = 0
        self.goal_hold_duration = 50
        self.reward_calc = RewardEvaluator(goal=self.goal)

        self.prev_x, self.prev_y, self.prev_time = None, None, None
        self.send_loop()

    def send_loop(self):
        data_log = []
        print("Starte Logging... (Strg+C zum Beenden)")

        try:
            while True:
                start = time.time()
                pose = self.vr.devices["tracker_1"].get_pose_euler()
                x, y, z = pose[0], pose[1], pose[2]
                yaw = pose[3]
                timestamp = time.time()

                v = 0.0
                if self.prev_x is not None:
                    dx_dt = x - self.prev_x
                    dy_dt = y - self.prev_y
                    dt = timestamp - self.prev_time
                    if dt > 0:
                        v = np.sqrt(dx_dt ** 2 + dy_dt ** 2) / dt
                self.prev_x, self.prev_y, self.prev_time = x, y, timestamp

                dx, dy = self.goal[0] - x, self.goal[1] - y
                distance = np.linalg.norm([dx, dy])
                angle_to_goal = np.arctan2(dy, dx)
                rel_angle = np.arctan2(np.sin(angle_to_goal - yaw), np.cos(angle_to_goal - yaw))

                state = np.array([dx, dy, distance, np.sin(rel_angle), np.cos(rel_angle), v], dtype=np.float32)
                state_tensor = torch.tensor(state).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    action = actor(state_tensor).cpu().numpy()[0]
                speed, steering = action
                speed = speed +0.6

                reward = self.reward_calc.compute_reward(np.array([x, y]), yaw, v)

                if distance < 0.2:
                    self.goal_hold_timer += 1
                    if self.goal_hold_timer >= self.goal_hold_duration:
                        reward += 100.0
                        self.goal = sample_goal(self.area_size)
                        self.goal_hold_timer = 0
                        self.reward_calc = RewardEvaluator(goal=self.goal)
                else:
                    self.goal_hold_timer = 0

                message = json.dumps({
                    "speed": float(speed),
                    "steering": float(steering)
                }) + "\n"
                self.sock.sendall(message.encode())

                entry = {
                    "timestamp": timestamp,
                    "position": {"x": x, "y": y},
                    "yaw": yaw,
                    "goal": self.goal.tolist(),
                    "state": state.tolist(),
                    "action": {"speed": float(speed), "steering": float(steering)},
                    "reward": float(reward)
                }
                data_log.append(entry)

                print(f"\rPos: ({x:.2f}, {y:.2f}) | V: {v:.2f} | R: {reward:.2f} | G: {self.goal} |  Speed and Steering: ({speed:.2f}, {steering:.2f})", end="")
                self.plot_position(x, y)

                sleep_time = self.interval - (time.time() - start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nBeende Logging...")
            self.sock.close()
            os.makedirs("plot_results/measurement_data", exist_ok=True)
            with open("plot_results/measurement_data/rl_tracker_live.json", "w") as f:
                json.dump(data_log, f, indent=2)
            print("✅ Daten gespeichert")

    def plot_position(self, x, y):
        self.x_data.append(x)
        self.y_data.append(y)

    def update_plot(self, frame):
        self.line.set_data(self.x_data, self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        return self.line,


if __name__ == '__main__':
    Client()
