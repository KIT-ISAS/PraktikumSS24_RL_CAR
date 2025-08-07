/*
  Arduino Controller Sketch

  This sketch toggles between RC and autonomous modes via a push-button,
  receives scaled speed and steering commands over serial ("speed,steering" CSV format),
  reads analog inputs when in RC mode, and outputs to an MCP4922 DAC via SPI.
  All logic and timing are preserved exactly; documentation has been enhanced in English.
*/

#include <SPI.h>

// === Pin Definitions ===
const int controller_Speed_Input    = A0;  // RC speed potentiometer input
const int controller_Steering_Input = A1;  // RC steering potentiometer input
const int buttonInput               = A2;  // Mode-toggle push-button input
const int DAC_CS_Pin                = 10;  // Chip select for MCP4922 DAC

// === Variables for incoming commands (voltage direct from Python) ===
float steeringV = 1.5;  // Current steering voltage command
float speedV    = 1.5;  // Current speed voltage command

// === Debounce state for push-button ===
int lastButtonState    = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;  // milliseconds

// === Mode flag: false = RC mode, true = Autonomous mode ===
bool useAutonomous = false;

// === Counters for automatic switching logic ===
int switchCounter = 0;  // Counts repeated "center" commands in autonomous mode
int rcCounter     = 0;  // Counts sustained RC inputs when returning to autonomous

void setup() {
  // Configure chip select and button input
  pinMode(DAC_CS_Pin, OUTPUT);
  pinMode(buttonInput, INPUT);

  // Initialize SPI bus
  SPI.begin();
  digitalWrite(DAC_CS_Pin, HIGH);

  // Initialize serial port for command input and debug output
  Serial.begin(19200);
  Serial.println("System ready. Waiting for mode selection ...");
}

void loop() {
  // --- 1) Mode toggle via push-button with debounce ---
  int currentButtonState = digitalRead(buttonInput);
  if (lastButtonState == HIGH && currentButtonState == LOW) {
    // Button pressed edge detected
    if ((millis() - lastDebounceTime) > debounceDelay) {
      lastDebounceTime = millis();
      useAutonomous = !useAutonomous;
      Serial.print("Mode changed: ");
      Serial.println(useAutonomous ? "AUTONOMOUS" : "RC");
    }
  }
  lastButtonState = currentButtonState;

  // --- 2) Read incoming serial data line-by-line ---
  static String input = "";
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      // Full line received in 'input'
      int commaIndex = input.indexOf(',');
      if (commaIndex > 0 && commaIndex < input.length() - 1) {
        // Extract CSV fields: steering,speed
        String steeringStr = input.substring(0, commaIndex);
        String speedStr    = input.substring(commaIndex + 1);

        float speed    = speedStr.toFloat();
        float steering = steeringStr.toFloat();

        // Only update if parsed values are valid floats
        if (!isnan(speed) && !isnan(steering)) {
          speedV    = speed;
          steeringV = steering;
        } else {
          Serial.println("Warning: invalid input ignored: " + input);
        }
      } else {
        Serial.println("Warning: malformed line received: " + input);
      }
      // Reset buffer for next line
      input = "";
    } else {
      // Accumulate characters up to safe limit
      if (input.length() < 50) {
        input += c;
      } else {
        input = "";
      }
    }
  }

  // === 3) Determine effective outputs based on mode ===
  float effSteeringV;
  float effSpeedV;

  if (useAutonomous) {
    // Autonomous: use voltage commands from serial
    effSteeringV = steeringV;
    effSpeedV    = speedV;
    // If both commands are exactly center (1.5V), count and switch back to RC
    if (effSpeedV == 1.5 && effSteeringV == 1.5) {
      switchCounter++;
      if (switchCounter >= 10) {
        rcCounter = 0;
        useAutonomous = false;
      }
    }
  } else {
    // RC mode: read analog controller inputs (0-5V)
    effSteeringV = analogRead(controller_Steering_Input) * (5.0 / 1023.0);
    effSpeedV    = analogRead(controller_Speed_Input)    * (5.0 / 1023.0);
    // If previously autonomous center count reached, count RC inputs to switch
    if (switchCounter >= 10) {
      rcCounter++;
      if (rcCounter > 10) {
        switchCounter = 0;
        useAutonomous = true;
      }
    }
  }

  // === 4) Constrain voltages and convert to 12-bit DAC values ===
  effSteeringV = constrain(effSteeringV, 0.0, 3.0);
  effSpeedV    = constrain(effSpeedV,    0.0, 3.0);

  int dacSteering = (int)((effSteeringV / 3.0) * 4095.0);
  int dacSpeed    = (int)((effSpeedV    / 3.0) * 4095.0);

  // Send to DAC channels
  writeMCP4922(0, dacSteering);
  writeMCP4922(1, dacSpeed);

  // Debug output
  Serial.print("Steering V: ");
  Serial.print(effSteeringV, 2);
  Serial.print("\tSpeed V: ");
  Serial.print(effSpeedV, 2);
  Serial.print("\tMode: ");
  Serial.println(useAutonomous ? "AUTONOMOUS" : "RC");
}

// Send a 12-bit value to the MCP4922 DAC via SPI
// channel: 0 for DAC A, 1 for DAC B
// value: 12-bit integer (0-4095)
void writeMCP4922(byte channel, int value) {
  value = constrain(value, 0, 4095);
  uint16_t command = 0b0011000000000000;  // DAC A settings: buffered, gain=1x, active
  if (channel == 1) command |= 0b1000000000000000;  // DAC B select
  command |= (value & 0x0FFF);

  digitalWrite(DAC_CS_Pin, LOW);
  SPI.beginTransaction(SPISettings(20000000, MSBFIRST, SPI_MODE0));
  SPI.transfer16(command);
  SPI.endTransaction();
  digitalWrite(DAC_CS_Pin, HIGH);
}
