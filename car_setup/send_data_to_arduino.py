# getControlData.py (auf dem Raspberry Pi)
import socket
import time
import serial

class Server:
    HOST = '0.0.0.0'  # Lauscht auf allen Interfaces
    PORT = 12345

    def __init__(self):
        # Serielle Verbindung mit Arduino
        try:
            self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
            print("Serielle Verbindung zum Arduino geöffnet.")
        except serial.SerialException as e:
            print(f"Fehler bei serieller Verbindung: {e}")
            self.ser = None

        # TCP Socket vorbereiten
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.HOST, self.PORT))    
        self.sock.listen()
        print(f"Warte auf Verbindung an Port {self.PORT}...")
        self.accept()

    def accept(self):
        conn, addr = self.sock.accept()
        print(f"TCP-Verbindung von {addr}")
        self.receive(conn)

    def receive(self, conn):
        while True:
            try:
                data = conn.recv(1024).decode().strip()
                if not data:
                    print("TCP-Verbindung geschlossen.")
                    break

                print("Empfangen:", data)

                # Weiterleiten an Arduino (z. B. "1.50,1.50\n")
                if self.ser and self.ser.is_open:
                    self.ser.write((data + "\n").encode())
                    print("→ Weitergeleitet an Arduino")

            except (ConnectionResetError, BrokenPipeError):
                print("TCP-Verbindung unterbrochen.")
                break

if __name__ == "__main__":
    Server()
