const int pinVibrationMotor = 3;
const String VIBRATION_ON_MESSAGE = "VIBRATION_ON";
const String VIBRATION_OFF_MESSAGE = "VIBRATION_OFF";

void setup() {
  // Initialize serial communication at 9600 bps
  Serial.begin(9600);
  
  // Set the vibration motor pin as OUTPUT
  pinMode(pinVibrationMotor, OUTPUT);
}

void loop() {
  // Check if there is data available on the serial port
  
  if (Serial.available() > 0) {
    // Read the message from the serial port
    String message = Serial.readStringUntil('\n');
    
    // Compare the received message
    if (message.equals("VIBRATION_ON_LOW")) {
      // Turn on the vibration motor at low level
      digitalWrite(pinVibrationMotor, 100);
      Serial.println("Vibration motor turned on at low level");
      delay(1000); // Wait for 1 second to avoid excessive vibration
    } else if (message.equals("VIBRATION_ON_HIGH")) {
      // Turn on the vibration motor at high level
      digitalWrite(pinVibrationMotor, HIGH);
      Serial.println("Vibration motor turned on at high level");
      delay(1000); // Wait for 1 second to avoid excessive vibration
    } else if (message.equals(VIBRATION_OFF_MESSAGE)) {
      // Turn off the vibration motor
      digitalWrite(pinVibrationMotor, LOW);
      Serial.println("Vibration motor turned off");
      delay(1000); // Wait for 1 second to avoid excessive vibration
    } else {
      // Print an error message for unrecognized message
      Serial.println("Error: Unrecognized message");
    }
  }
}
