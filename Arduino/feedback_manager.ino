enum vibrationMode {
  NONE,
  MOTORW1,
  MOTORW2,
  MOTORB1,
  MOTORB2,
  MOTORW,
  MOTORB,
};

#define motorpin_w1 3
#define motorpin_w2 5
#define motorpin_b1 6
#define motorpin_b2 7
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

// Peltier variables
const int pCells = 2;
int actualPower[pCells] = {0,0}; // Power level fro 0 to 99%
int powerToReach[pCells] = {0,0};
int peltierLevel[pCells]; // This is a value from 0 to 255 that actually controls the MOSFET

// Vibration variables
const int vMotors = 4;
vibrationMode vMode = NONE;
int actualLevel[vMotors] = {0,0,0,0}; // Intensity level fro 0 to 99%
int levelToReach[vMotors] = {0,0,0,0};
int vibrationLevel[vMotors]; // This is a value from 0 to 255 that actually controls the MOSFET

void setup()
{
  for (int i = 0; i < pCells; i++)
  {
    peltierLevel[i] = toAnalog(actualPower[i]);
  }
  for (int i = 0; i < vMotors; i++)
  {
    vibrationLevel[i] = toAnalog(actualLevel[i]);
  }
  
  Serial.begin(9600);
  
  for (int i = 0; i < pCells; i++)
  {
    printPeltierState(i);
  }
  for (int i = 0; i < vMotors; i++)
  {
    printVibrationState(i);
  }
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
      powerToReach[value] = correctValue(intensity);
    } else if (!strcmp(code,"v")) {
      vMode = toVibrationMode(value);
      if (vMode == NONE) {
        for (int i = 0; i < vMotors; i++)
        {
          levelToReach[i] = 0;
        }
      } else if (vMode == MOTORW1) {
        levelToReach[0] = correctValue(intensity);
      } else if (vMode == MOTORW2) {
        levelToReach[1] = correctValue(intensity);
      } else if (vMode == MOTORB1) {
        levelToReach[2] = correctValue(intensity);
      } else if (vMode == MOTORB2) {
        levelToReach[3] = correctValue(intensity);
      } else if (vMode == MOTORW) {
        levelToReach[0] = correctValue(intensity);
        levelToReach[1] = correctValue(intensity);
      } else if (vMode == MOTORB) {
        levelToReach[2] = correctValue(intensity);
        levelToReach[3] = correctValue(intensity);
      }
    }
    
    newData = false;
  }
  
  for (int i = 0; i < pCells; i++)
  {
    if (actualPower[i] != powerToReach[i]) {
      if (actualPower[i] < powerToReach[i]) {
        actualPower[i] += 2;
      } else {
        actualPower[i] -= 2;
      }
      
      peltierLevel[i] = toAnalog(actualPower[i]);
      changePeltier(i, peltierLevel[i]);
      
      printPeltierState(i);
      if (actualPower[i] == powerToReach[i]) {
        Serial.println("");
      }
      delay(500);
    }
  }
  
  for (int i = 0; i < vMotors; i++)
  {
    if (actualLevel[i] != levelToReach[i]) {
      /*if (actualLevel[i] < levelToReach[i]) {
        actualLevel[i] += 2;
      } else {
        actualLevel[i] -= 2;
      }*/
      actualLevel[i] = levelToReach[i];
      
      vibrationLevel[i] = toAnalog(actualLevel[i]);
      changeVibration(i, vibrationLevel[i]);
      
      printPeltierState(i);
      if (actualLevel[i] == levelToReach[i]) {
        Serial.println("");
      }
      //delay(500);
    }
  }
}

//============

vibrationMode toVibrationMode(int mode) {
  vibrationMode actualMode = NONE;
  
  switch(mode) {
    case 1:
      actualMode = MOTORW1;
      break;
    case 2:
      actualMode = MOTORW2;
      break;
    case 3:
      actualMode = MOTORB1;
      break;
    case 4:
      actualMode = MOTORB2;
      break;
    case 5:
      actualMode = MOTORW;
      break;
    case 6:
      actualMode = MOTORB;
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

int toAnalog(int value) {
  return map(value, 0, 99, 0, 255);
}

void changePeltier(int code, int power) {
  switch (code)
  {
  case 0:
    analogWrite(peltier_0, peltierLevel[code]); //Write this new value out to the port
    break;
  case 1:
    analogWrite(peltier_1, peltierLevel[code]); //Write this new value out to the port
    break;
  }
}

void changeVibration(int code, int power) {
  switch (code)
  {
  case 0:
    analogWrite(motorpin_w1, vibrationLevel[code]); //Write this new value out to the port
    break;
  case 1:
    analogWrite(motorpin_w2, vibrationLevel[code]); //Write this new value out to the port
    break;
  case 2:
    analogWrite(motorpin_b1, vibrationLevel[code]); //Write this new value out to the port
    break;
  case 3:
    analogWrite(motorpin_b2, vibrationLevel[code]); //Write this new value out to the port
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

void printVibrationState(int code) {
  Serial.print("VibrationCode=");
  Serial.print(code);
  Serial.print("Level=");
  Serial.print(actualLevel[code]);
  if (actualLevel[code] != levelToReach[code]) {
    Serial.print(" LevelToReach=");
    Serial.print(levelToReach[code]);
  }
  Serial.print(" VLevel=");
  Serial.println(vibrationLevel[code]);
}

int correctValue(int value) {
  if (value < 0) {
    return 0;
  } else if (value > 99) {
    return 99;
  } else {
    return value;
  }
}