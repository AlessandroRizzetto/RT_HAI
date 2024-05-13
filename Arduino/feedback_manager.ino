#define motorpin_wl 3
#define motorpin_wr 5
#define motorpin_bl 6
#define motorpin_br 9
#define peltier_neck 10
#define peltier_chest 11

// DATA DEFINITIONs

// Vibration motors to start
enum VibrationMode
{
  NONE,
  MOTORW1, // Wrist 1
  MOTORW2, // Wrist 2
  MOTORB1, // Back 1
  MOTORB2, // Back 2
  MOTORW,  // Both wrists
  MOTORB,  // Both backs
};

// Pattern to follow to reach the desired intensity/ies
enum PatternMode
{
  LINEAR,     // Reach the desired level (average of min_intensity and max_intensity) and then stop
  SINUSOIDAL, // Keep changing level from min_intensity to max_intensity and vice versa
};

// Device class to handle the devices
class Device
{
private:
  int pace;
  PatternMode mode;
  int min_intensity;
  int max_intensity;
  int actualValue; // Power level fro 0 to 99%
  int valueToReach;
  float powerMultiplier; // Multiplier to adapt different power sources or different devices characteristics

  int computePace(int diff, int defaultPace);

public:
  Device(PatternMode mode, float powerMultiplier, int min_intensity, int max_intensity, int pace);
  Device(PatternMode mode, float powerMultiplier, int pace);

  float getPowerMultiplier();
  int getPace();
  PatternMode getMode();
  int getMinIntensity();
  int getMaxIntensity();
  int getActualValue();
  int getValueToReach();
  int getMultipliedValue();
  int getAnalogValue();

  void setPace(int pace);
  void setMode(PatternMode mode);
  void setMinIntensity(int min_intensity);
  void setMaxIntensity(int max_intensity);
  void setDevice(PatternMode mode, int min_intensity, int max_intensity);

  boolean updateDevice();
  static int correctValue(int value);
};

// Serial Data parsing
const byte numChars = 16;
char receivedChars[numChars];
char tempChars[numChars]; // Temporary array for use when parsing
// Variables to hold the parsed data
char code[2];                 // ["h","v"] Group of devices
int value = 0;                // [0-1 for "h", 0-5 for "v"] Group-specific device(s)
PatternMode pattern = LINEAR; // Type of progress to the desired intensity level
int min_intensity = 0;        // [0-99] Min intensity level to reach
int max_intensity = 0;        // [0-99] Max intensity level to reach
int pace = 0;                 // [0-99] level progression per cycle
boolean newData = false;

// Peltier variables
const int pCells = 2;
const int pPace = 5;
Device pHandlers[pCells] = {Device(LINEAR, 1, pPace), Device(LINEAR, 1, pPace)}; // List of objects which keep track and controll Peltier devices

// Vibration variables
const int vMotors = 4;
const int vPace = 5;
VibrationMode vMode = NONE;
Device vHandlers[vMotors] = {Device(LINEAR, 1, vPace), Device(LINEAR, 1, vPace), Device(LINEAR, 1, vPace), Device(LINEAR, 1, vPace)}; // List of objects which keep track and controll Vibration motors

int devicesIndex[pCells + vMotors] = {-1, -1, -1, -1, -1, -1}; // List of device indexes at which to change its level
int count;

// FUNCTIONS PROTOTYPES

// Device class constructor

Device::Device(PatternMode mode, float powerMultiplier, int min_intensity, int max_intensity, int pace)
{
  this->powerMultiplier = powerMultiplier;
  this->setDevice(mode, min_intensity, max_intensity);
  this->setPace(pace);
  this->actualValue = 0;
}

Device::Device(PatternMode mode, float powerMultiplier, int pace) : Device(mode, powerMultiplier, 0, 0, pace) {}

// Device class getters

float Device::getPowerMultiplier()
{
  return powerMultiplier;
}

int Device::getPace()
{
  return pace;
}

PatternMode Device::getMode()
{
  return mode;
}

int Device::getMinIntensity()
{
  return min_intensity;
}

int Device::getMaxIntensity()
{
  return max_intensity;
}

int Device::getActualValue()
{
  return actualValue;
}

int Device::getValueToReach()
{
  return valueToReach;
}

int Device::getMultipliedValue()
{
  return round(actualValue * powerMultiplier);
}

int Device::getAnalogValue()
{
  return map(getMultipliedValue(), 0, 99, 0, 255);
}

// Device class setters

void Device::setPace(int pace)
{
  if (pace > 0)
  {
    this->pace = pace;
  }
  else
  {
    this->pace = max(abs(this->valueToReach - this->actualValue), this->max_intensity - this->min_intensity);
  }
}

void Device::setMode(PatternMode mode)
{
  this->mode = mode;
}

void Device::setMinIntensity(int min_intensity)
{
  this->min_intensity = min_intensity;
}

void Device::setMaxIntensity(int max_intensity)
{
  this->max_intensity = max_intensity;
}

// Set the device with the desired pattern and intensity, deriving the other parameters
void Device::setDevice(PatternMode mode, int min_intensity, int max_intensity)
{
  this->mode = mode;
  this->min_intensity = min(min_intensity, max_intensity);
  this->max_intensity = max(min_intensity, max_intensity);

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

// Update the device intensity
boolean Device::updateDevice()
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

// Correct the value to be between 0 and 99
int Device::correctValue(int value)
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

// Compute the pace to reach the desired intensity
int Device::computePace(int diff, int defaultPace)
{
  int actualPace = min(abs(diff), defaultPace);
  return (diff >= 0) ? actualPace : -actualPace;
}

// Convert a given code to the actual vibration mode
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

// Convert a given code to the actual pattern mode
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

// Read the serial data, remove "<" and ">" and store it in the receivedChars array
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

// Split the data into its parts
void parseData() // Message format: <code,value,pattern,min_intensity,max_intensity,pace>
{
  char *strtokIndx; // This is used by strtok() as an index

  strtokIndx = strtok(tempChars, ",");
  strcpy(code, strtokIndx);

  strtokIndx = strtok(NULL, ",");
  value = atoi(strtokIndx);

  strtokIndx = strtok(NULL, ",");
  pattern = toPatternMode(atoi(strtokIndx));

  strtokIndx = strtok(NULL, ",");
  min_intensity = atoi(strtokIndx);

  strtokIndx = strtok(NULL, ",");
  max_intensity = atoi(strtokIndx);

  strtokIndx = strtok(NULL, ",");
  pace = atoi(strtokIndx);
}

// Change the peltier power
void changePeltier(int code, int power)
{
  switch (code)
  {
  case 0:
    analogWrite(peltier_neck, power); // Write this new value out to the port
    break;
  case 1:
    analogWrite(peltier_chest, power);
    break;
  }
}

// Change the vibration power
void changeVibration(int code, int power)
{
  switch (code)
  {
  case 0:
    analogWrite(motorpin_wl, power); // Write this new value out to the port
    break;
  case 1:
    analogWrite(motorpin_wr, power);
    break;
  case 2:
    analogWrite(motorpin_bl, power);
    break;
  case 3:
    analogWrite(motorpin_br, power);
    break;
  }
}

// Print the peltier state
void printPeltierState(int code)
{
  Serial.print("PeltierCode=");
  Serial.print(code);
  Serial.print(" Mode=");
  Serial.print(pHandlers[code].getMode());
  Serial.print(" Power=");
  Serial.print(pHandlers[code].getActualValue());
  Serial.print(" MultipliedLevel=");
  Serial.print(pHandlers[code].getMultipliedValue());

  if (pHandlers[code].getActualValue() != pHandlers[code].getValueToReach())
  {
    Serial.print(" PToReach=");
    Serial.print(pHandlers[code].getValueToReach());
  }

  Serial.print(" Pace=");
  Serial.println(pHandlers[code].getPace());
}

// Print the vibration state
void printVibrationState(int code)
{
  Serial.print("VibrationCode=");
  Serial.print(code);
  Serial.print(" Mode=");
  Serial.print(vHandlers[code].getMode());
  Serial.print(" Level=");
  Serial.print(vHandlers[code].getActualValue());
  Serial.print(" MultipliedLevel=");
  Serial.print(vHandlers[code].getMultipliedValue());

  if (vHandlers[code].getActualValue() != vHandlers[code].getValueToReach())
  {
    Serial.print(" LToReach=");
    Serial.print(vHandlers[code].getValueToReach());
  }

  Serial.print(" Pace=");
  Serial.println(vHandlers[code].getPace());
}

// Arduino setup and loop functions
void setup()
{
  Serial.begin(9600);

  pinMode(motorpin_wl, OUTPUT);
  pinMode(motorpin_wr, OUTPUT);
  pinMode(motorpin_bl, OUTPUT);
  pinMode(motorpin_br, OUTPUT);

  pinMode(peltier_neck, OUTPUT);
  pinMode(peltier_chest, OUTPUT);

  // Peltier setup
  for (int i = 0; i < pCells; i++)
  {
    printPeltierState(i);
  }

  // Vibration setup
  for (int i = 0; i < vMotors; i++)
  {
    printVibrationState(i);
  }
}

void loop()
{
  recvWithStartEndMarkers();
  if (newData == true) // If new data is available, parse it
  {
    strcpy(tempChars, receivedChars);
    // Temporary copy necessary to protect the original data (strtok() in parseData() replaces the commas with \0)
    parseData();
    min_intensity = Device::correctValue(min_intensity);
    max_intensity = Device::correctValue(max_intensity);

    if (!strcmp(code, "h")) // If the code is "h", set peltier cells with the parsed values
    {
      pHandlers[value].setDevice(pattern, min_intensity, max_intensity);
      pHandlers[value].setPace(pace);
    }
    else if (!strcmp(code, "v")) // If the code is "v", set vibration motors with the parsed values
    {
      vMode = toVibrationMode(value);
      switch (vMode) // Set the devices index to update based on the mode
      {
      case NONE:
        for (int i = 0; i < vMotors; i++)
        {
          devicesIndex[i] = i;
        }
        break;
      case MOTORW1:
        devicesIndex[0] = 0;
        break;
      case MOTORW2:
        devicesIndex[0] = 1;
        break;
      case MOTORB1:
        devicesIndex[0] = 2;
        break;
      case MOTORB2:
        devicesIndex[0] = 3;
        break;
      case MOTORW:
        for (int i = 0; i < 2; i++)
        {
          devicesIndex[i] = i;
        }
        break;
      case MOTORB:
        for (int i = 2; i < 4; i++)
        {
          devicesIndex[i - 2] = i;
        }
        break;
      }
    }

    // Set the devices with the parsed values
    count = 0;
    while (devicesIndex[count] != -1 && count < pCells + vMotors)
    {
      vHandlers[devicesIndex[count]].setDevice(pattern, min_intensity, max_intensity);
      vHandlers[devicesIndex[count]].setPace(pace);
      devicesIndex[count] = -1;
      count++;
    }

    newData = false;
  }

  // Update the devices
  // Peltier update
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

  // Vibration update
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