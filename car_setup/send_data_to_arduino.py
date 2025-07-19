# send_data_to_arduino.py
import socket
import time
import serial
import json

class Server:
    HOST = '0.0.0.0'
    PORT = 12345

    def __init__(self):
        # Serielle Verbindung zum Arduino
        try:
            self.ser = serial.Serial('/dev/ttyACM0', 19200, timeout=1)
            time.sleep(2)
            print("✅ Serielle Verbindung zum Arduino geöffnet.")
        except serial.SerialException as e:
            print(f"❌ Fehler bei serieller Verbindung: {e}")
            self.ser = None

        # TCP-Server vorbereiten
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.HOST, self.PORT))
        self.sock.listen(1)
        print(f"🔌 Warte auf Verbindung an Port {self.PORT}...")

        self.accept_loop()

    def accept_loop(self):
        while True:
            conn, addr = self.sock.accept()
            print(f"🔗 TCP-Verbindung von {addr}")
            try:
                self.handle_client(conn)
            except Exception as e:
                print(f"⚠️ Fehler während Verbindung: {e}")
            finally:
                conn.close()
                print("❌ Verbindung geschlossen.\nWarte auf neue Verbindung...")

    def handle_client(self, conn):
        buffer = ""
        last_line = None
        conn.settimeout(1.0)

        while True:
            try:
                data = conn.recv(1024)
                if not data:
                    break
                buffer += data.decode()

                # Hol nur die letzte vollständige Zeile
                while "\n" in buffer:
                    line, buffer = buffer.rsplit("\n", 1)
                    lines = line.strip().split("\n")
                    if lines:
                        last_line = lines[-1].strip()

                if last_line:
                    self.process_json_line(last_line)
                    last_line = None  # danach zurücksetzen

            except socket.timeout:
                continue
            except Exception as e:
                print(f"⚠️ Datenfehler: {e}")
                break


    def process_json_line(self, line):
        try:
            msg = json.loads(line)
            speed = float(msg.get("speed", 0.0))
            steering = float(msg.get("steering", 0.0))
            # print(f"JSON empfangen: speed={speed}, steering={steering}")

            # Skalieren
            scaled_speed = 1.5 + max(-0.3, min(0.3, speed * 0.3))
            scaled_steering = 1.5 + steering * 1.5

            # Sende als CSV (nicht als JSON)
            serial_msg = f"{scaled_speed:.2f},{scaled_steering:.2f}\n"
            print(f"→ Sende an Arduino (CSV): {serial_msg.strip()}")

            if self.ser and self.ser.is_open:
                self.ser.write(serial_msg.encode())

        except json.JSONDecodeError:
            print(f"❌ Ungültige JSON-Zeile: {line}")
        except Exception as e:
            print(f"⚠️ Fehler beim Verarbeiten: {e}")

if __name__ == "__main__":
    Server()
