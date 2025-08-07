"""
send_data_to_arduino.py

TCP server receiving JSON-formatted control commands and forwarding them as
scaled CSV messages over serial to an Arduino.
"""

# === Imports ===
import socket
import time
import serial
import json


class Server:
    """
    TCP server that listens for JSON messages containing 'speed' and 'steering',
    scales and converts them to CSV format, and sends over a serial connection
to an attached Arduino.
    """

    HOST = '0.0.0.0'  # Listen on all network interfaces
    PORT = 12345      # TCP port to bind

    def __init__(self):
        """
        Initialize serial connection to Arduino and start accepting TCP clients.
        """
        # Attempt to open serial port to Arduino
        try:
            self.ser = serial.Serial('/dev/ttyACM0', 19200, timeout=1)
            time.sleep(2)  # Allow Arduino to reset
            print("Serial connection to Arduino opened.")
        except serial.SerialException as e:
            print(f"Error opening serial connection: {e}")
            self.ser = None

        # Prepare TCP server socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.HOST, self.PORT))
        self.sock.listen(1)
        print(f"Waiting for TCP connection on port {self.PORT}...")

        # Enter accept loop
        self.accept_loop()

    def accept_loop(self):
        """
        Continuously accept incoming TCP connections and handle them.
        """
        while True:
            conn, addr = self.sock.accept()
            print(f"TCP connection established from {addr}")
            try:
                self.handle_client(conn)
            except Exception as e:
                print(f"Error during connection: {e}")
            finally:
                conn.close()
                print("Connection closed. Waiting for new connection...")

    def handle_client(self, conn):
        """
        Receive data from a TCP client, extract the latest complete JSON line,
        and process it.

        Args:
            conn: socket object for the connected client
        """
        buffer = ""
        last_line = None
        conn.settimeout(1.0)

        while True:
            try:
                data = conn.recv(1024)
                if not data:
                    break  # Client closed connection
                buffer += data.decode()

                # Retrieve only the last complete line
                while "\n" in buffer:
                    line, buffer = buffer.rsplit("\n", 1)
                    lines = line.strip().split("\n")
                    if lines:
                        last_line = lines[-1].strip()

                # Process the most recent line if available
                if last_line:
                    self.process_json_line(last_line)
                    last_line = None

            except socket.timeout:
                continue  # No data received, continue listening
            except Exception as e:
                print(f"Data error: {e}")
                break

    def process_json_line(self, line):
        """
        Parse a JSON line containing 'speed' and 'steering', scale values, and
        send as CSV to the Arduino.

        Args:
            line: str containing a JSON object
        """
        try:
            # Parse JSON and extract float values
            msg = json.loads(line)
            speed = float(msg.get("speed", 0.0))
            steering = float(msg.get("steering", 0.0))

            # Scale values:
            #   speed factor: base 1.5 ± 0.3 * input
            #   steering factor: base 1.5 ± 1.5 * input
            scaled_speed = 1.5 + max(-0.3, min(0.3, speed * 0.3))
            scaled_steering = 1.5 + steering * 1.5

            # Format as CSV "speed,steering" with two decimals
            serial_msg = f"{scaled_speed:.2f},{scaled_steering:.2f}\n"
            print(f"Sending to Arduino (CSV): {serial_msg.strip()}")

            # Send over serial if available
            if self.ser and self.ser.is_open:
                self.ser.write(serial_msg.encode())

        except json.JSONDecodeError:
            print(f"Invalid JSON line: {line}")
        except Exception as e:
            print(f"Error processing line: {e}")


if __name__ == "__main__":
    # Create and run the server
    Server()
