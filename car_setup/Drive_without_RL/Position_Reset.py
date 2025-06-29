import numpy as np
import time
import sys
import json
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import socket

try:
    import triad_openvr
    USE_VR = True
except ImportError:
    print("⚠️ triad_openvr nicht gefunden – Simuliere Tracker-Daten.")
    USE_VR = False


class Client:
    def __init__(self, raspberry_host="192.168.137.5", raspberry_port=12345):
        self.interval = 1 / 50  # 20 Hz
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((raspberry_host, raspberry_port))
            print(f"✅ Verbunden mit Raspberry Pi ({raspberry_host}:{raspberry_port})")
        except Exception as e:
            print(f"❌ Verbindung fehlgeschlagen: {e}")
            sys.exit(1)

        if USE_VR:
            self.vr = triad_openvr.triad_openvr()
            self.vr.print_discovered_objects()

        self.x_data, self.y_data = [], []
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100)

        self.goal = np.array([-0.3, -0.3])      # Fixes Ziel
        self.goal_reached = False               # Zustand merken
        self.send_loop()

    def send_loop(self):
        data_log = []
        print("Starte Steuerung... (Strg+C zum Beenden)")

        try:
            while True:
                start = time.time()

                # === Position und Orientierung ===
                if USE_VR:
                    pose = self.vr.devices["tracker_1"].get_pose_euler()
                    x, y = pose[0], pose[1]
                    yaw = pose[3]
                else:
                    x, y, yaw = self.simulate_motion()

                dx, dy = self.goal[0] - x, self.goal[1] - y
                distance = np.linalg.norm([dx, dy])
                angle_to_goal = np.arctan2(dy, dx)
                rel_angle = np.arctan2(np.sin(angle_to_goal - yaw), np.cos(angle_to_goal - yaw))

                min_working_speed = 0.85
                max_speed = 0.9
                max_steering = 0.6

                # === Spielfeld-Grenzen prüfen ===
                if not (-2.0 <= x <= 2.0 and -2.0 <= y <= 2.0):
                    print(f"\n🛑 Außerhalb des Spielfelds! Pos: ({x:.2f}, {y:.2f}) – STOPP!")
                    speed = 0.0
                    steering = 0.0

                elif self.goal_reached:
                    speed = 0.0
                    steering = 0.0

                elif distance < 0.1:
                    speed = 0.0
                    steering = 0.0
                    self.goal_reached = True
                    print("\n🎯 Ziel erreicht. Stoppe dauerhaft.")

                else:
                    speed = min(min_working_speed, max_speed)

                    # Adaptive Lenkung: sanfter je näher am Ziel
                    steering_gain = min(1.0, distance / 1.5)
                    steering = np.clip(steering_gain * rel_angle, -max_steering, max_steering)

                # === Nachricht senden ===
                message = json.dumps({
                    "speed": float(speed),
                    "steering": float(steering)
                }) + "\n"
                self.sock.sendall(message.encode())

                print(f"\rPos: ({x:.2f}, {y:.2f}) | Ziel: {self.goal} | ⏩ Speed: {speed:.2f}, Steering: {steering:.2f}", end="")
                self.plot_position(x, y)

                sleep_time = self.interval - (time.time() - start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n🛑 Stoppe...")
            self.sock.close()
            os.makedirs("plot_results/measurement_data", exist_ok=True)
            with open("plot_results/measurement_data/non_rl_drive_log.json", "w") as f:
                json.dump(data_log, f, indent=2)
            print("✅ Daten gespeichert")

    def simulate_motion(self):
        # Dummy-Daten ohne VR
        t = time.time()
        x = 1.0 + 0.5 * np.sin(t / 2)
        y = 1.0 + 0.5 * np.cos(t / 2)
        yaw = np.arctan2(np.cos(t / 2), np.sin(t / 2))
        return x, y, yaw

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
