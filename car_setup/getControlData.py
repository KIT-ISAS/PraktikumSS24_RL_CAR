# getControlData.py (auf dem Raspberry Pi)
import socket
import time

class Server:
    HOST = '0.0.0.0'  # lauscht auf allen Interfaces
    PORT = 12345      

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.HOST, self.PORT))    
        self.sock.listen()
        print("Warte auf Verbindung...")
        self.accept()

    def accept(self):
        conn, addr = self.sock.accept()
        print(f"Verbindung von {addr}")
        self.receive(conn)

    def receive(self, conn):
        while True:
            start = time.time()
            data = conn.recv(1024).decode()
            if not data:
                print("Verbindung geschlossen.")
                break
            print("Empfangen:", data)
            # hier könntest du Motorsteuerung einbauen
            print(time.time() - start)

if __name__ == "__main__":
    Server()
