import tkinter as tk
import serial
import time

# === Serielle Verbindung öffnen (Pfad ggf. anpassen)
ser = serial.Serial('/dev/ttyACM1', 9600, timeout=1)
time.sleep(2)  # Warten auf Arduino-Reset

def send_values():
    steering = (steering_scale.get() + 1) * 1.5
    speed = (speed_scale.get() + 1) * 1.5
    msg = f"{steering:.2f},{speed:.2f}\n"
    ser.write(msg.encode())
    root.after(100, send_values)

root = tk.Tk()
root.title("RC Overrides")

tk.Label(root, text="Steering (-1 to 1)").pack()
steering_scale = tk.Scale(root, from_=-1.0, to=1.0, resolution=0.01,
                          orient=tk.HORIZONTAL, length=300)
steering_scale.pack()

tk.Label(root, text="Speed (-1 to 1)").pack()
speed_scale = tk.Scale(root, from_=-1.0, to=1.0, resolution=0.01,
                       orient=tk.HORIZONTAL, length=300)
speed_scale.pack()

root.after(100, send_values)
root.mainloop()
