#define motorpin_wl 3
#define motorpin_wr 5
#define motorpin_bl 6
#define motorpin_br 9
#define peltier_neck 10
#define peltier_chest 11

enum VibrationMode
{
  NONE,
  MOTORW1,
  MOTORW2,
  MOTORB1,
  MOTORB2,
  MOTORW,
  MOTORB,
};

enum PatternMode
{
  LINEAR,
  SINUSOIDAL,
};

class Device
{
private:
  int pace;
  PatternMode mode;
  int min_intensity;
  int max_intensity;
  int actualValue; // Power level fro 0 to 99%
  int valueToReach;
  int computePace(int diff, int defaultPace)
  {

    int actualPace;

    if (diff >= 0)
    {
      if (diff >= defaultPace)
      {
        actualPace = defaultPace;
      }
      else
      {
        actualPace = diff;
      }
    }
    else
    {
      if ((-diff) >= defaultPace)
      {
        actualPace = -defaultPace;
      }
      else
      {
        actualPace = diff;
      }
    }

    return actualPace;
  }

public:
  void setDevice(PatternMode mode, int min_intensity, int max_intensity)
  {
    this->mode = mode;
    this->min_intensity = min_intensity;
    this->max_intensity = max_intensity;

    switch (this->mode)
    {
    case LINEAR:
      this->valueToReach = Device::correctValue((min_intensity + max_intensity) / 2);
      break;
    case SINUSOIDAL:
      this->valueToReach = max_intensity;
      break;
    }
  }
  Device(PatternMode mode, int min_intensity, int max_intensity, int pace)
  {
    this->setDevice(mode, min_intensity, max_intensity);
    this->setPace(pace);
    this->actualValue = 0;
  }
  Device(PatternMode mode, int pace)
  {
    this->setDevice(mode, 0, 0);
    this->setPace(pace);
    this->actualValue = 0;
  }
  static int correctValue(int value)
  {
    if (value < 0)
    {
      return 0;
    }
    else if (value > 99)
    {
      return 99;
    }
    else
    {
      return value;
    }
  }
  int getPace()
  {
    return pace;
  }
  PatternMode getMode()
  {
    return mode;
  }
  int getMinIntensity()
  {
    return min_intensity;
  }
  int getMaxIntensity()
  {
    return max_intensity;
  }
  int getActualValue()
  {
    return actualValue;
  }
  int getValueToReach()
  {
    return valueToReach;
  }
  int getAnalogValue()
  {
    return map(actualValue, 0, 99, 0, 255);
  }
  void setPace(int pace)
  {
    if (pace > 0)
    {
      this->pace = pace;
    }
    else
    {
      this->pace = abs(this->valueToReach - this->actualValue);
    }
  }
  void setMode(PatternMode mode)
  {
    this->mode = mode;
  }
  void setMinIntensity(int min_intensity)
  {
    this->min_intensity = min_intensity;
  }
  void setMaxIntensity(int max_intensity)
  {
    this->max_intensity = max_intensity;
  }
  boolean updateDevice()
  {
    boolean updated = false;

    if (this->actualValue != this->valueToReach)
    {
      switch (this->mode)
      {
      case LINEAR:
        this->actualValue += computePace(this->valueToReach - this->actualValue, this->pace);
        updated = true;
        break;
      case SINUSOIDAL:
        this->actualValue += computePace(this->valueToReach - this->actualValue, this->pace);

        if (this->actualValue == this->valueToReach)
        {
          if (this->valueToReach == this->max_intensity)
          {
            this->valueToReach = this->min_intensity;
          }
          else
          {
            this->valueToReach = this->max_intensity;
          }
        }
        updated = true;
        break;
      }
    }

    return updated;
  }
};

// Serial Data parsing
const byte numChars = 16;
char receivedChars[numChars];
char tempChars[numChars]; // temporary array for use when parsing
// variables to hold the parsed data
char code[2];
int value = 0;
PatternMode pattern = LINEAR;
int min_intensity = 0;
int max_intensity = 0;
int pace = 0;
boolean newData = false;

// Peltier variables
const int pCells = 2;
const int pPace = 5;
Device pHandlers[pCells] = {Device(LINEAR, pPace), Device(LINEAR, pPace)};

// Vibration variables
const int vMotors = 4;
const int vPace = 5;
VibrationMode vMode = NONE;
Device vHandlers[vMotors] = {Device(LINEAR, vPace), Device(LINEAR, vPace), Device(LINEAR, vPace), Device(LINEAR, vPace)};

void setup()
{
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
  recvWithStartEndMarkers(); // Format: <code,value,min_intensity,max_intensity,pattern>
  if (newData == true)
  {
    strcpy(tempChars, receivedChars);
    // this temporary copy is necessary to protect the original data
    //   because strtok() used in parseData() replaces the commas with \0
    parseData();
    min_intensity = Device::correctValue(min_intensity);
    max_intensity = Device::correctValue(max_intensity);
    if (!strcmp(code, "h"))
    {
      pHandlers[value].setDevice(pattern, min_intensity, max_intensity);
      pHandlers[value].setPace(pace);
    }
    else if (!strcmp(code, "v"))
    {
      vMode = toVibrationMode(value);
      switch (vMode)
      {
      case NONE:
        for (int i = 0; i < vMotors; i++)
        {
          vHandlers[i].setDevice(pattern, min_intensity, max_intensity);
          vHandlers[i].setPace(pace);
        }
        break;
      case MOTORW1:
        vHandlers[0].setDevice(pattern, min_intensity, max_intensity);
        vHandlers[0].setPace(pace);
        break;
      case MOTORW2:
        vHandlers[1].setDevice(pattern, min_intensity, max_intensity);
        vHandlers[1].setPace(pace);
        break;
      case MOTORB1:
        vHandlers[2].setDevice(pattern, min_intensity, max_intensity);
        vHandlers[2].setPace(pace);
        break;
      case MOTORB2:
        vHandlers[3].setDevice(pattern, min_intensity, max_intensity);
        vHandlers[3].setPace(pace);
        break;
      case MOTORW:
        vHandlers[0].setDevice(pattern, min_intensity, max_intensity);
        vHandlers[1].setDevice(pattern, min_intensity, max_intensity);
        for (int i = 0; i < 2; i++)
        {
          vHandlers[i].setPace(pace);
        }
        break;
      case MOTORB:
        vHandlers[2].setDevice(pattern, min_intensity, max_intensity);
        vHandlers[3].setDevice(pattern, min_intensity, max_intensity);
        for (int i = 2; i < 4; i++)
        {
          vHandlers[i].setPace(pace);
        }
        break;
      }
    }

    newData = false;
  }

  for (int i = 0; i < pCells; i++)
  {
    if (pHandlers[i].updateDevice())
    {
      changePeltier(i, pHandlers[i].getAnalogValue());

      printPeltierState(i);
      if (pHandlers[i].getActualValue() == pHandlers[i].getValueToReach())
      {
        Serial.println("");
      }
      delay(500);
    }
  }

  for (int i = 0; i < vMotors; i++)
  {
    if (vHandlers[i].updateDevice())
    {
      changeVibration(i, vHandlers[i].getAnalogValue());

      printVibrationState(i);
      if (vHandlers[i].getActualValue() == vHandlers[i].getValueToReach())
      {
        Serial.println("");
      }
      delay(500);
    }
  }
}

//============

VibrationMode toVibrationMode(int mode)
{
  VibrationMode actualMode = NONE;

  switch (mode)
  {
  case 0:
    actualMode = MOTORW1;
    break;
  case 1:
    actualMode = MOTORW2;
    break;
  case 2:
    actualMode = MOTORB1;
    break;
  case 3:
    actualMode = MOTORB2;
    break;
  case 4:
    actualMode = MOTORW;
    break;
  case 5:
    actualMode = MOTORB;
    break;
  }

  return actualMode;
}

PatternMode toPatternMode(int mode)
{
  PatternMode actualMode = LINEAR;

  switch (mode)
  {
  case 0:
    actualMode = LINEAR;
    break;
  case 1:
    actualMode = SINUSOIDAL;
    break;
  }

  return actualMode;
}

void recvWithStartEndMarkers()
{

  static boolean recvInProgress = false;
  static byte ndx = 0;
  char startMarker = '<';
  char endMarker = '>';
  char rc;

  while (Serial.available() > 0 && newData == false)
  {
    rc = Serial.read();
    if (recvInProgress == true)
    {
      if (rc != endMarker)
      {
        receivedChars[ndx] = rc;
        ndx++;
        if (ndx >= numChars)
        {
          ndx = numChars - 1;
        }
      }
      else
      {
        receivedChars[ndx] = '\0'; // terminate the string
        recvInProgress = false;
        ndx = 0;
        newData = true;
      }
    }
    else if (rc == startMarker)
    {
      recvInProgress = true;
    }
  }
}

//============

void parseData() // Message format: <code,value,pattern,min_intensity,max_intensity,pace>
{                // split the data into its parts

  char *strtokIndx; // this is used by strtok() as an index

  strtokIndx = strtok(tempChars, ","); // get the first part - the string
  strcpy(code, strtokIndx);            // copy it to dir

  strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
  value = atoi(strtokIndx);       // convert this part to an integer

  strtokIndx = strtok(NULL, ",");
  pattern = toPatternMode(atoi(strtokIndx)); // convert this part to an integer

  strtokIndx = strtok(NULL, ",");
  min_intensity = atoi(strtokIndx); // convert this part to an integer

  strtokIndx = strtok(NULL, ",");
  max_intensity = atoi(strtokIndx); // convert this part to an integer

  strtokIndx = strtok(NULL, ",");
  pace = atoi(strtokIndx); // convert this part to an integer
}

void changePeltier(int code, int power)
{
  switch (code)
  {
  case 0:
    analogWrite(peltier_neck, power); // Write this new value out to the port
    break;
  case 1:
    analogWrite(peltier_chest, power); // Write this new value out to the port
    break;
  }
}

void changeVibration(int code, int power)
{
  switch (code)
  {
  case 0:
    analogWrite(motorpin_wl, power); // Write this new value out to the port
    break;
  case 1:
    analogWrite(motorpin_wr, power); // Write this new value out to the port
    break;
  case 2:
    analogWrite(motorpin_bl, power); // Write this new value out to the port
    break;
  case 3:
    analogWrite(motorpin_br, power); // Write this new value out to the port
    break;
  }
}

void printPeltierState(int code)
{
  Serial.print("PeltierCode=");
  Serial.print(code);
  Serial.print(" Mode=");
  Serial.print(pHandlers[code].getMode());
  Serial.print(" Power=");
  Serial.print(pHandlers[code].getActualValue());

  if (pHandlers[code].getActualValue() != pHandlers[code].getValueToReach())
  {
    Serial.print(" PToReach=");
    Serial.print(pHandlers[code].getValueToReach());
  }

  Serial.print(" Pace=");
  Serial.println(pHandlers[code].getPace());
}

void printVibrationState(int code)
{
  Serial.print("VibrationCode=");
  Serial.print(code);
  Serial.print(" Mode=");
  Serial.print(vHandlers[code].getMode());
  Serial.print(" Level=");
  Serial.print(vHandlers[code].getActualValue());

  if (vHandlers[code].getActualValue() != vHandlers[code].getActualValue())
  {
    Serial.print(" LToReach=");
    Serial.print(pHandlers[code].getValueToReach());
  }

  Serial.print(" Pace=");
  Serial.println(vHandlers[code].getPace());
}