#include <SPI.h>

// === Pinbelegung === 
const int controller_Speed_Input = A0;
const int controller_Steering_Input = A1;
const int buttonInput = A2;
const int DAC_CS_Pin = 10;

// === Vorgaben aus Python (direkt in Volt, 0.0 – 3.0)
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

  Serial.begin(9600);
  Serial.println("System bereit. Warte auf Moduswahl ...");
}

void loop() {
  // === Modus per Taster ändern (fallende Flanke)
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

  // === Seriellen Input lesen (bereits skaliert auf 0–3 V)
  if (Serial.available()) {
  String input = Serial.readStringUntil('\n');
  int commaIndex = input.indexOf(',');
  if (commaIndex > 0) {
    String sStr = input.substring(0, commaIndex);       // Speed
    String vStr = input.substring(commaIndex + 1);      // Steering
    float s = sStr.toFloat();
    float v = vStr.toFloat();
    if ((s != 0.0 || sStr.startsWith("0")) && (v != 0.0 || vStr.startsWith("0"))) {
      speedV = constrain(s, 0.0, 3.0);
      steeringV = constrain(v, 0.0, 3.0);
    }
  }
}

  float effectiveSteeringV, effectiveSpeedV;
  if (useAutonomous) {
    effectiveSteeringV = steeringV;
    effectiveSpeedV = speedV;
  } else {
    effectiveSteeringV = analogRead(controller_Steering_Input) * (5.0 / 1023.0);
    effectiveSpeedV = analogRead(controller_Speed_Input) * (5.0 / 1023.0);
  }

  // Begrenzen auf max. VREF 3.0 V
  effectiveSteeringV = constrain(effectiveSteeringV, 0.0, 3.0);
  effectiveSpeedV = constrain(effectiveSpeedV, 0.0, 3.0);

  // Umrechnen auf DAC-Werte (0–4095)
  int dacSteering = (effectiveSteeringV / 3.0) * 4095.0;
  int dacSpeed = (effectiveSpeedV / 3.0) * 4095.0;

  // DAC-Ausgabe
  writeMCP4922(0, dacSteering);  // VOUTA = Steering
  writeMCP4922(1, dacSpeed);     // VOUTB = Speed

  // Debug
  Serial.print("Steering V: ");
  Serial.print(effectiveSteeringV, 2);
  Serial.print("\tSpeed V: ");
  Serial.print(effectiveSpeedV, 2);
  Serial.print("\tModus: ");
  Serial.println(useAutonomous ? "AUTONOM" : "RC");

  delay(50);
}

void writeMCP4922(byte channel, int value) {
  value = constrain(value, 0, 4095);
  uint16_t command = 0b0011000000000000;  // Gain = 1, VREF unbuffered, SHDN=1
  if (channel == 1) command |= 0b1000000000000000;
  command |= (value & 0x0FFF);

  digitalWrite(DAC_CS_Pin, LOW);
  SPI.beginTransaction(SPISettings(20000000, MSBFIRST, SPI_MODE0));
  SPI.transfer16(command);
  SPI.endTransaction();
  digitalWrite(DAC_CS_Pin, HIGH);
}
