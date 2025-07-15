#include <SPI.h>

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

int switchCounter = 0;
int rcCounter = 0;

void setup() {
  pinMode(DAC_CS_Pin, OUTPUT);
  pinMode(buttonInput, INPUT);

  SPI.begin();
  digitalWrite(DAC_CS_Pin, HIGH);

  Serial.begin(19200);
  Serial.println("System bereit. Warte auf Moduswahl ...");
}

void loop() {
  // === Modus-Umschaltung über Taster
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

  // === Eingehende Daten auslesen
  // Eingehende Daten zeichenweise lesen
static String input = "";
while (Serial.available()) {
  char c = Serial.read();

  if (c == '\n') {
    // Komplette Zeile empfangen
    int commaIndex = input.indexOf(',');
    if (commaIndex > 0 && commaIndex < input.length() - 1) {
      String steeringStr = input.substring(0, commaIndex);
      String speedStr = input.substring(commaIndex + 1);

      float speed = speedStr.toFloat();
      float steering = steeringStr.toFloat();

      if (!isnan(speed) && !isnan(steering)) {
        speedV = speed;
        steeringV = steering;
      } else {
        Serial.println("⚠️ Ungültige Eingabe ignoriert: " + input);
      }
    } else {
      Serial.println("⚠️ Fehlerhafte Zeile empfangen: " + input);
    }

    input = "";  // Zeilenpuffer zurücksetzen
  } else {
    if (input.length() < 50) {  // Sicherheit: max. Länge
      input += c;
    } else {
      input = "";  // Wenn zu lang: verwerfen
    }
  }
}



  float effSteeringV, effSpeedV;

  if (useAutonomous) {
    effSteeringV = steeringV;
    effSpeedV = speedV;
    if (effSpeedV == 1.5 && effSteeringV == 1.5)
    {
      switchCounter++;
      if (switchCounter >= 10)
      {
        rcCounter = 0;
        useAutonomous = false;
      }
      
    }
    
  } else {
    effSteeringV = analogRead(controller_Steering_Input) * (5.0 / 1023.0);
    effSpeedV = analogRead(controller_Speed_Input) * (5.0 / 1023.0);
    if (switchCounter >= 10)
    {
      rcCounter++;
      if (rcCounter > 10)
      {
        switchCounter = 0;
        useAutonomous = true;
      }
      
    }
    
  }

  effSteeringV = constrain(effSteeringV, 0.0, 3.0);
  effSpeedV = constrain(effSpeedV, 0.0, 3.0);

  int dacSteering = (effSteeringV / 3.0) * 4095.0;
  int dacSpeed = (effSpeedV / 3.0) * 4095.0;

  writeMCP4922(0, dacSteering);
  writeMCP4922(1, dacSpeed);

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
  if (channel == 1) command |= 0b1000000000000000;
  command |= (value & 0x0FFF);

  digitalWrite(DAC_CS_Pin, LOW);
  SPI.beginTransaction(SPISettings(20000000, MSBFIRST, SPI_MODE0));
  SPI.transfer16(command);
  SPI.endTransaction();
  digitalWrite(DAC_CS_Pin, HIGH);
}
