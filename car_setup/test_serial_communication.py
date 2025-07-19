import tkinter as tk
import serial
import time

# Serielle Verbindung öffnen
try:
    ser = serial.Serial('/dev/ttyACM0', 19200, timeout=1)
    time.sleep(2)
    print("✅ Serielle Verbindung zum Arduino geöffnet.")
except serial.SerialException as e:
    print(f"❌ Fehler bei serieller Verbindung: {e}")
    ser = None

def send_values():
    steering = steering_var.get()
    speed = speed_var.get()
    print(f"Raw Slider: Speed={speed}, Steering={steering}")

    # Skalierung: -1..1 → Arduino-Format
    scaled_speed = 1.5 + max(-0.2, min(0.2, speed * 0.2))      # → 1.3 .. 1.7
    scaled_steering = 1.5 + steering * 1.5                    # → 0.0 .. 3.0

    msg = f"{scaled_speed:.2f},{scaled_steering:.2f}\n"
    print(f"→ Sende an Arduino: {msg.strip()}")

    if ser and ser.is_open:
        try:
            ser.write(msg.encode())
        except Exception as e:
            print(f"⚠️ Fehler beim Schreiben an Serial: {e}")

    root.after(100, send_values)

# GUI
root = tk.Tk()
root.title("RC Overrides")

steering_var = tk.DoubleVar()
speed_var = tk.DoubleVar()

tk.Label(root, text="Steering (-1 to 1)").pack()
steering_scale = tk.Scale(root, from_=-1.0, to=1.0, resolution=0.01,
                          orient=tk.HORIZONTAL, length=300, variable=steering_var)
steering_scale.pack()

tk.Label(root, text="Speed (-1 to 1)").pack()
speed_scale = tk.Scale(root, from_=-1.0, to=1.0, resolution=0.01,
                       orient=tk.HORIZONTAL, length=300, variable=speed_var)
speed_scale.pack()

root.after(100, send_values)
root.mainloop()
