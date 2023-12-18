enum vibrationMode {
  NONE,
  MOTOR1,
  MOTOR2,
  MOTOR3,
  MOTOR12,
  MOTOR13,
  MOTOR23,
};

#define motorpin1 3
#define motorpin2 5
#define motorpin3 6
#define peltier 9

// Serial Data parsing
const byte numChars = 16;
char receivedChars[numChars];
char tempChars[numChars];        // temporary array for use when parsing
// variables to hold the parsed data
char code[2];
int value = 0;
boolean newData = false;

// Vibration variables
vibrationMode vMode = NONE;
vibrationMode actualVMode = NONE;

int power = 0; // Power level fro 0 to 99%
int powerToReach = 0;
int peltierLevel = map(power, 0, 99, 0, 255); // This is a value from 0 to 255 that actually controls the MOSFET

void setup()
{
  pinMode(motorpin1, OUTPUT);
  pinMode(motorpin2, OUTPUT);
  pinMode(motorpin3, OUTPUT);
  
  Serial.begin(9600);
  
  Serial.print("Power=");
  Serial.print(power);
  Serial.print(" PLevel=");
  Serial.println(peltierLevel);
  Serial.print("Mode=");
  Serial.println(vMode);
  Serial.println("");
}

//============

void loop()
{
  recvWithStartEndMarkers(); //Format: <code,value>
  if (newData == true) {
    strcpy(tempChars, receivedChars);
    // this temporary copy is necessary to protect the original data
    //   because strtok() used in parseData() replaces the commas with \0
    parseData();
    if (!strcmp(code,"h")) {
      powerToReach = value;
      
      if(powerToReach > 99) powerToReach = 99;
      if(powerToReach < 0) powerToReach = 0;
    } else if (!strcmp(code,"v")) {
      vMode = toVibrationMode(value);
    }
    
    newData = false;
  }
  
  if (power != powerToReach) {
    if (power < powerToReach) {
      power += 2;
    } else {
      power -= 2;
    }
    
    peltierLevel = map(power, 0, 99, 0, 255);
    analogWrite(peltier, peltierLevel); //Write this new value out to the port
    
    Serial.print("Power=");
    Serial.print(power);
    Serial.print(" PowerToReach=");
    Serial.print(powerToReach);
    Serial.print(" PLevel=");
    Serial.println(peltierLevel);
    if (power == powerToReach) {
      Serial.println("");
    }
    delay(500);
  }
  
  if (vMode != actualVMode) {
    actualVMode = vMode;
    if (actualVMode == NONE) {
      digitalWrite(motorpin1, LOW);
      digitalWrite(motorpin2, LOW);
    } else if (actualVMode == MOTOR1) {
      digitalWrite(motorpin1, HIGH);
      digitalWrite(motorpin2, LOW);
    } else if (actualVMode == MOTOR2) {
      digitalWrite(motorpin1, LOW);
      digitalWrite(motorpin2, HIGH);
    } else if (actualVMode == MOTOR12) {
      digitalWrite(motorpin1, HIGH);
      digitalWrite(motorpin2, HIGH);
    }
    
    Serial.print("Mode=");
    Serial.println(vMode);
    Serial.println("");
  }
}

//============

vibrationMode toVibrationMode(int mode) {
  vibrationMode actualMode = NONE;
  
  switch(mode) {
    case 1:
      actualMode = MOTOR1;
      break;
    case 2:
      actualMode = MOTOR2;
      break;
    case 3:
      actualMode = MOTOR3;
      break;
    case 4:
      actualMode = MOTOR12;
      break;
    case 5:
      actualMode = MOTOR13;
      break;
    case 6:
      actualMode = MOTOR23;
      break;
  }
  
  return actualMode;
}

void recvWithStartEndMarkers() {
  
  static boolean recvInProgress = false;
  static byte ndx = 0;
  char startMarker = '<';
  char endMarker = '>';
  char rc;
  
  while (Serial.available() > 0 && newData == false) {
    rc = Serial.read();
    if (recvInProgress == true) {
      if (rc != endMarker) {
        receivedChars[ndx] = rc;
        ndx++;
        if (ndx >= numChars) {
          ndx = numChars - 1;
        }
      } else {
        receivedChars[ndx] = '\0'; // terminate the string
        recvInProgress = false;
        ndx = 0;
        newData = true;
      }
    } else if (rc == startMarker) {
      recvInProgress = true;
    }
  }
}

//============

void parseData() { // split the data into its parts

  char * strtokIndx; // this is used by strtok() as an index

  strtokIndx = strtok(tempChars, ",");     // get the first part - the string
  strcpy(code, strtokIndx); // copy it to dir

  strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
  value = atoi(strtokIndx);     // convert this part to an integer, speed
}