
void setup() {
  Serial.begin(9600);
  Serial.println("Arduino has been activated. Starting Serial communication...");
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(PIR_PIN_INPUT, INPUT);     
  // TO DO : Activate the sense function

}

void loop() {
  delay(500);
  sendData();
  
  
  if(Serial.available()>0){
    String data = Serial.readStringUntil('\n'); //Reading the first line until there is a backspace
    
    
  }

}

void sendData(){
  // Define variables for sensor readings

  //...

  // Check if any reads failed and exit early (to try again).
  // if (""value"") {
  //   Serial.println(F("Failed to read from sensor!"));
  //   return;
  // }
  
  // Compute heat index in Celsius (isFahreheit = false)
  float hic = dht.computeHeatIndex(t, h, false);
  
  Serial.println((String) "HUM"+h);
  Serial.println((String) "TEM"+t);
  Serial.println((String) "HIC"+hic);

// TO PRINT ON THE SERIAL THE FUNCTION IS: Serial.println((String) "..."+...);
// ISNAN IS A FUNCTION THAT CHECKS IF THE VALUE IS A NUMBER OR NOT
}