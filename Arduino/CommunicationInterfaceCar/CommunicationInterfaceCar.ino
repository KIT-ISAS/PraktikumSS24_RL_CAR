#include <SPI.h>
#include <ArduinoJson.h>

// === Pinbelegung === 
const int controller_Speed_Input = A0;
const int controller_Steering_Input = A1;
const int buttonInput = A2;
const int DAC_CS_Pin = 10;

// === Vorgaben aus Python (direkt in Volt)
float steeringV = 1.5;
float speedV = 1.5;

// Tasterentprellung
int lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

// Modus-Umschaltung: false = RC, true = Autonom
bool useAutonomous = false;

void setup() {
  pinMode(DAC_CS_Pin, OUTPUT);
  pinMode(buttonInput, INPUT);

  SPI.begin();
  digitalWrite(DAC_CS_Pin, HIGH);

  Serial.begin(115200);  // ← schnellere Übertragung
  Serial.println("System bereit. Warte auf Moduswahl ...");
}

void loop() {
  // === Taster-Modusumschaltung (fallende Flanke)
  int currentButtonState = digitalRead(buttonInput);
  if (lastButtonState == HIGH && currentButtonState == LOW) {
    if ((millis() - lastDebounceTime) > debounceDelay) {
      lastDebounceTime = millis();
      useAutonomous = !useAutonomous;
      Serial.print("Modus gewechselt: ");
      Serial.println(useAutonomous ? "AUTONOM" : "RC");
    }
  }
  lastButtonState = currentButtonState;

  // === Seriellen JSON-Input lesen (vom Pi)
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    StaticJsonDocument<100> doc;
    DeserializationError err = deserializeJson(doc, input);
    if (!err) {
      float s = doc["speed"];
      float v = doc["steering"];
      // Skaliere von -1 .. 1 → 0 .. 3 V
      speedV = constrain((s + 1.0) * 1.5, 1.3, 1.7);     // Nur speed begrenzen!
      steeringV = (v + 1.0) * 1.5;
    } else {
      Serial.println("JSON-Parsing-Fehler");
    }
  }

  float effSteeringV, effSpeedV;

  if (useAutonomous) {
    effSteeringV = steeringV;
    effSpeedV = speedV;
    Serial.print("Python vorgabe: ");
    Serial.print(effSteeringV);
    Serial.print(" / ");
    Serial.println(effSpeedV);
  } else {
    effSteeringV = analogRead(controller_Steering_Input) * (5.0 / 1023.0);
    effSpeedV = analogRead(controller_Speed_Input) * (5.0 / 1023.0);
  }

  // DAC darf nur bis 3.0 V ausgeben
  effSteeringV = constrain(effSteeringV, 0.0, 3.0);
  effSpeedV = constrain(effSpeedV, 0.0, 3.0);

  // DAC-Werte berechnen
  int dacSteering = (effSteeringV / 3.0) * 4095.0;
  int dacSpeed = (effSpeedV / 3.0) * 4095.0;

  // DAC-Ausgabe
  writeMCP4922(0, dacSteering);  // VOUTA = Steering
  writeMCP4922(1, dacSpeed);     // VOUTB = Speed

  // Debug
  Serial.print("Steering V: ");
  Serial.print(effSteeringV, 2);
  Serial.print("\tSpeed V: ");
  Serial.print(effSpeedV, 2);
  Serial.print("\tModus: ");
  Serial.println(useAutonomous ? "AUTONOM" : "RC");
}

void writeMCP4922(byte channel, int value) {
  value = constrain(value, 0, 4095);
  uint16_t command = 0b0011000000000000;
  if (channel == 1) command |= 0b1000000000000000;  // Kanal B
  command |= (value & 0x0FFF);

  digitalWrite(DAC_CS_Pin, LOW);
  SPI.beginTransaction(SPISettings(20000000, MSBFIRST, SPI_MODE0));
  SPI.transfer16(command);
  SPI.endTransaction();
  digitalWrite(DAC_CS_Pin, HIGH);
}
