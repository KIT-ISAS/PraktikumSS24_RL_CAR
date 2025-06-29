import json
import matplotlib.pyplot as plt
import os

# Datei einlesen
data_dir = 'measurement_data/'
filename = f"{data_dir}rl_tracker_live.json"

if not os.path.exists(filename):
    print(f"❌ Datei nicht gefunden: {filename}")
    exit(1)

with open(filename, "r") as f:
    data = json.load(f)

# Daten extrahieren
x_vals = [entry["position"]["x"] for entry in data]
y_vals = [entry["position"]["y"] for entry in data]
speed_vals = [entry.get("speed_calculated", 0.0) for entry in data]

# Plot als Heatmap (Scatter mit Farbskala)
plt.figure(figsize=(8, 6))
sc = plt.scatter(x_vals, y_vals, c=speed_vals, cmap='viridis', s=10, edgecolors='none')
plt.colorbar(sc, label='Geschwindigkeit (m/s)')
plt.title("Heatmap der Positionen nach berechneter Geschwindigkeit")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.axis('equal')
plt.grid(True)

plt.tight_layout()
plt.show()
