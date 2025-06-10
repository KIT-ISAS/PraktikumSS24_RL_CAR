import triad_openvr
import time
import sys
import socket
import json
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

v = triad_openvr.triad_openvr()
v.print_discovered_objects()

if len(sys.argv) == 1:
    interval = 1/250
elif len(sys.argv) == 2:
    interval = 1/float(sys.argv[1])
else:
    print("Invalid number of arguments")
    interval = False

# Ziel- und Startkoordinaten im Dateinamen
start_x, start_y = 0, 0
end_x, end_y = 2, 2
outpt_dir = 'plot_results/measurement_data/'
output_file = f"{outpt_dir}tracker_data_start_{start_x}_{start_y}_end_{end_x}_{end_y}.json"

# Datenliste vorbereiten
data_log = []

print("Starte Logging... (Strg+C zum Beenden)")

class Client:
    HOST = "192.168.137.5"
    PORT = 12345

    def __init__(self):
        #self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.sock.connect((self.HOST, self.PORT))

        # Init for plotting
        self.x_data = []
        self.y_data = []

        # Start plot animation
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100)

        # Start data loop
        self.send_loop()

    def send_loop(self):
        prev_x, prev_y, prev_time = None, None, None
        try:
            while True:
                start = time.time()
                txt = ""
                data = v.devices["tracker_1"].get_pose_euler()

                x, y, z = data[0], data[1], data[2]
                yaw, pitch, roll = data[3], data[4], data[5]
                timestamp = time.time()

                # Placeholder für spätere Felder
                speed = None
                steering = None

                # Geschwindigkeit berechnen (nur x und y)
                if prev_x is not None and prev_y is not None and prev_time is not None:
                    dx = x - prev_x
                    dy = y - prev_y
                    dt = timestamp - prev_time

                    if dt > 0:
                        speed_calculated = (dx ** 2 + dy ** 2) ** 0.5 / dt
                    else:
                        speed_calculated = 0.0
                else:
                    speed_calculated = 0.0

                # Aktuelle Werte speichern für nächsten Durchlauf
                prev_x, prev_y, prev_time = x, y, timestamp

                # Eintrag hinzufügen
                entry = {
                    "timestamp": timestamp,
                    "position": {"x": x, "y": y, "z": z},
                    "rotation": {"yaw": yaw, "pitch": pitch, "roll": roll},
                    "speed": speed,
                    "steering": steering,
                    "speed_calculated": speed_calculated
                }

                data_log.append(entry)

                # Format and print data
                txt = " ".join(["%.4f" % val for val in data])
                print("\r" + txt, end="")

                # Send data
                #self.sock.sendall(txt.encode())

                # Update plot data
                self.plot_position(data)

                # Sleep to maintain interval
                sleep_time = interval - (time.time() - start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nSpeichere Daten in Datei...")

            # Sicher speichern
            with open(output_file, "w") as f:
                json.dump(data_log, f, indent=2)

            print(f"✅ Daten erfolgreich gespeichert in: {output_file}")

    def plot_position(self, pose_data):
        x = pose_data[0]
        y = pose_data[1]
        yaw = pose_data[3]  # Assuming yaw is at index 3

        # Optional: apply yaw transformation to movement direction here
        # (currently not adjusting direction by yaw)

        self.x_data.append(x)
        self.y_data.append(y)

    def update_plot(self, frame):
        self.line.set_data(self.x_data, self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        return self.line,

if __name__ == '__main__':
    Client()
