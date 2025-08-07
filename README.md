# RC Car Navigation with Reinforcement Learning

## Projektübersicht
Dieses Projekt steuert ein ferngesteuertes Auto (RC Car) mittels Reinforcement Learning (RL). Positionsdaten werden über HTC Vive Tracker erfasst und per Raspberry Pi an ein Arduino-Interface gesendet, das Motor und Lenkung antreibt.

## Repository-Struktur
```
PraktikumSS24_RL_CAR/
├── Arduino/
│   └── CommunicationInterfaceCar/
│       └── CommunicationInterfaceCar.ino        # Arduino-Sketch
├── car_setup/
│   ├── RL_main.py                              # RL-Kontrollskript im Echtbetrieb
│   ├── send_data_to_arduino.py                 # TCP-Server, leitet Daten an Arduino weiter
│   └── plot_results/                           # Logs und Modellcheckpoints
│       └── plot_jsonData.py
├── simulation/
│   ├── DDPG_RL_PyTorch.py                      # Offline-Training/Evaluation DDPG
│   ├── SAC_RL_PyTorch.py                       # Offline-Training/Evaluation SAC
│   └── envs/
│       └── rc_car_env.py                       # Gym-Environment
├── docs/
│   ├── schematics/                             # Schaltpläne (PDF/PNG)
│   └── network_setup.md                        # WLAN- und SSH-Konfiguration
├── Bewertung/                                  # Bewertungsunterlagen
│   ├── Bewertung_Praktikum.ods
│   └── Bewertung_Vortrag.xlsx
├── Presentations/                              # Präsentationen (PDF & PPTX)
├── Abschlussbericht_RL_basierte_Navigation_auf_offener_Fläche_mit_HTC_Vive_gestützter_Positionsbestimmung.pdf
├── Docker-ROS.pdf
├── GPU-Server-Manual.pdf
└── README.md                                   # Diese Datei
```

## Entry Points
- **Echtbetrieb**
  - `car_setup/RL_main.py`: Führt Echtzeit-Tracking & RL-Steuerung aus (Holodeck-Labor -> pi). 
  - `car_setup/send_data_to_arduino.py`: TCP-Server auf Raspberry Pi, leitet JSON (speed, steering) als CSV an Arduino weiter (pi -> Arduino).
- **Simulation & Training**
  - `simulation/DDPG_RL_PyTorch.py`: DDPG-Agent trainieren oder evaluieren.
  - `simulation/SAC_RL_PyTorch.py`: SAC-Agent trainieren oder evaluieren.
- **Hardware-Interface**
  - `Arduino/CommunicationInterfaceCar/CommunicationInterfaceCar.ino`: Steuert DAC-Ausgänge für Motor und Lenkung.

## Hardware-Setup
1. **HTC Vive Tracker & Base-Stations**: Im Holodeck-Labor installieren, mit SteamVR kalibrieren.
2. **Netzwerk**: Labor-Computer erstellt WLAN-Hotspot, Raspberry Pi verbindet sich automatisch.
3. **SSH**: `ssh pi@<IP-des-Pi>` (Benutzer: `pi`, Passwort: `raspberry`).

### Schaltpläne
Alle Schaltpläne findest du im Ordner `docs/schematics/`:
- `rc_car_schaltplan.pdf`
- `vive_tracker_network.png`

## Board Flashen
- **Arduino Nano Every**: Boardtyp `arduino:mega:avr:nonaevery` wählen.  
- **Flashing via Arduino CLI**: arduino-cli board list

- **Raspberry Pi**: Raspberry Pi OS auf SD-Karte flashen (z.B. mit Raspberry Pi Imager), über SSH konfigurieren.

## Software-Setup
### Voraussetzungen
- Python 3.8 oder neuer
- Bibliotheken: `torch`, `numpy`, `pygame`, `gymnasium`, `triad_openvr`, `pyserial`
- SteamVR (für Tracking)

### Installation
```bash
pip install torch numpy pygame gymnasium triad_openvr pyserial
```

## Nutzung
### Echtbetrieb
```bash
# 1. Raspberry Pi
cd PraktikumSS24_RL_CAR/car_setup
python3 send_data_to_arduino.py

# 2. Labor-Computer
cd PraktikumSS24_RL_CAR/car_setup
python3 RL_main.py
```

### Simulation & Training
```bash
cd PraktikumSS24_RL_CAR/simulation
# DDPG
python3 DDPG_RL_PyTorch.py
# SAC
python3 SAC_RL_PyTorch.py
```

## Weitere Dokumentation
- **Projektbericht**: `Abschlussbericht_RL_basierte_Navigation_auf_offener_Fläche_mit_HTC_Vive_gestützter_Positionsbestimmung.pdf`
- **Docker & ROS**: `Docker-ROS.pdf`
- **GPU-Server**: `GPU-Server-Manual.pdf`

*Entwickelt im Rahmen des Praktikum SS25 am ISAS, KIT.*  
