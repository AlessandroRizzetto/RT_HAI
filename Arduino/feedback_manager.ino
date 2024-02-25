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
#define peltier_0 9
#define peltier_1 10

// Serial Data parsing
const byte numChars = 16;
char receivedChars[numChars];
char tempChars[numChars];        // temporary array for use when parsing
// variables to hold the parsed data
char code[2];
int value = 0;
int intensity = 0;
boolean newData = false;

// Vibration variables
vibrationMode vMode = NONE;
vibrationMode actualVMode = NONE;

int actualPower[2] = {0,0}; // Power level fro 0 to 99%
int powerToReach[2] = {0,0};
int peltierLevel[2] = {toPeltier(actualPower[0]),toPeltier(actualPower[1])}; // This is a value from 0 to 255 that actually controls the MOSFET

void setup()
{
  pinMode(motorpin_w1, OUTPUT);
  pinMode(motorpin_w2, OUTPUT);
  pinMode(motorpin_b1, OUTPUT);
  pinMode(motorpin_b2, OUTPUT);
  
  Serial.begin(9600);
  
  for (int i = 0; i < 2; i++)
  {
    printPeltierState(i);
  }
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
      powerToReach[value] = intensity;
      
      if(powerToReach[value] > 99) powerToReach[value] = 99;
      if(powerToReach[value] < 0) powerToReach[value] = 0;
    } else if (!strcmp(code,"v")) {
      vMode = toVibrationMode(value);
    }
    
    newData = false;
  }
  
  for (int i = 0; i < 2; i++)
  {
    if (actualPower[i] != powerToReach[i]) {
      if (actualPower[i] < powerToReach[i]) {
        actualPower[i] += 2;
      } else {
        actualPower[i] -= 2;
      }
      
      peltierLevel = toPeltier(actualPower[i]);
      changePeltier(i, peltierLevel[i])
      
      printPeltierState(i);
      if (actualPower[i] == powerToReach[i]) {
        Serial.println("");
      }
      delay(500);
    }
  }
  
  
  if (vMode != actualVMode) {
    actualVMode = vMode;
    if (actualVMode == NONE) {
      digitalWrite(motorpin_w1, LOW);
      digitalWrite(motorpin_w2, LOW);
    } else if (actualVMode == MOTOR1) {
      digitalWrite(motorpin_w1, HIGH);
      digitalWrite(motorpin_w2, LOW);
    } else if (actualVMode == MOTOR2) {
      digitalWrite(motorpin_w1, LOW);
      digitalWrite(motorpin_w2, HIGH);
    } else if (actualVMode == MOTOR13) {
      digitalWrite(motorpin_w1, HIGH);
      digitalWrite(motorpin_w2, HIGH);
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
      actualMode = MOTOR13;
      break;
    case 5:
      actualMode = MOTOR23;
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
  value = atoi(strtokIndx);     // convert this part to an integer

  strtokIndx = strtok(NULL, ",");
  intensity = atoi(strtokIndx);     // convert this part to an integer
}

int toPeltier(int value) {
  return map(value, 0, 99, 0, 255);
}

void changePeltier(int code, int power) {
  switch (code)
  {
  case 0:
    analogWrite(peltier_0, peltierLevel[i]); //Write this new value out to the port
    break;
  case 1:
    analogWrite(peltier_1, peltierLevel[i]); //Write this new value out to the port
    break;
  }
}

void printPeltierState(int code) {
  Serial.print("PeltierCode=");
  Serial.print(code);
  Serial.print(" Power=");
  Serial.print(actualPower[code]);
  if (actualPower[code] != powerToReach[code]) {
    Serial.print(" PowerToReach=");
    Serial.print(powerToReach[code]);
  }
  Serial.print(" PLevel=");
  Serial.println(peltierLevel[code]);
}