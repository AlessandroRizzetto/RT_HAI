<p align="center">
  <h2 align="center">Human-Agent Interaction <br> for Awareness in Self-Presentations</h2>
  <p align="center">Alessandro Rizzetto - Simone Compri</p>
</p>
<br>

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Project Description](#project-description)
- [Software Architecture](#software-architecture)
  - [Installation](#installation)
  - [Usage](#usage)
- [Hardware Feedback System](#hardware-feedback-system)
- [Acknowledgments](#acknowledgments)

# Project Description

Welcome to the Human-Agent Interaction for Awareness in Self-Presentations project! This research initiative, conducted at the University of Trento, focuses on real-time analysis of user behavior during communication with a VR virtual agent.

The primary goal is to examine user-agent interactions in both real-time and offline modes. This analysis aims to provide valuable insights into human-agent dynamics, contributing to the enhancement of user self-presentation skills.

# Software Architecture

This software meticulously analyzes user movements and audio signals during interactions with the virtual agent. Data extraction from a webcam and microphone enables the extraction of features, which can be observed in real-time or analyzed offline.

The following software components were integral to the project:

- **Social Signal Interpretation (SSI) Framework:**

  - SSI framework provides tools to record, analyze, and recognize real-time human behavior, including gestures, mimics, head nods, and emotional speech. Its role in this project is crucial for collecting diverse sources, ensuring synchronization, and compensating for potential missing data.

- **Mediapipe:**

  - Mediapipe is employed for extracting data from a webcam, facilitating real-time analysis of user movements and gestures.

    <img src="https://camo.githubusercontent.com/54e5f06106306c59e67acc44c61b2d3087cc0a6ee7004e702deb1b3eb396e571/68747470733a2f2f6d65646961706970652e6465762f696d616765732f6d6f62696c652f706f73655f747261636b696e675f66756c6c5f626f64795f6c616e646d61726b732e706e67" width="500" height="300" />

- **OpenSmile:**
  - OpenSmile is utilized for processing audio data, extracting features from the user's speech, contributing to the analysis of emotional content and vocal nuances.

## Installation

This project was developed for `Python 3.10`.

To install the necessary dependencies for this project, follow the following steps:

1. Install Python requirements

   ```bash
   pip install -r requirements.txt
   ```

2. Set up SSI environment
   ```bash
   ./install.cmd
   ```

## Usage

The software has two different modes of operation: real-time and offline. The former is used to analyze user behavior during interactions with the virtual agent, while the latter is employed for offline analysis of pre-recorded data.
At the end of both analysis, the software will generate a CSV file containing the extracted data and prints a overall summary of the features.

In addition, differents routines of configurations are available to set the environment for every user.

The entire software can be executed using a GUI, which provides all the main functioalities. To execute it, run the following command:

```bash
python3 gui.py
```

![GUI](/media/gui.jpg)

In case of audio analysis, once starded the GUI, the first thing to do is to `Start Feedback System`, clicking the proper button. This will setup the environment necessary for the software to run.
An additional setup that could be done for audio analysis consist in manuallly editing configurations prameters of files `.pipeline-config` in `audio\pipes`.

After that the user can choose which functionality run among the following.

### OFFLINE ANALYSIS

1. VIDEO (Start: press `Start Video Analysis`. End: automatic): perform an analysis of the video specified in the `Path to video` field. Otherwise it will perform an analysis on a default video.

2. AUDIO (Start: press `Start Audio Analysis`. End: click on the terminal and press `Enter` on keyboard): perform an audio analysis similarly to offline video analysis.
   - VAD configuration (Start: press `Start VAD configuration`, `1` on the terminal when asked. End: close threshold window, click on the terminal and press `Enter` on keyboard): beginning this procedure it is possible to set the threshold for VAD. Namely, the selected audio is played, the new window on the left show the original audio, the right one the VADed audio and, through the slider in the central window, it is possible to change the the threshold, seeing the results on the right window.

### ONLINE ANALYSIS

1. VIDEO (Start: press `Start Video Analysis`. End: click on the video window and press `q` on keyboard): perform an analysis of live stream from the computer webcam.

   - Body configuration (Start: press `Start Body configuration`. End: click on the video window and press `q` on keyboard): this procedure allows the user to calibrate data for online video analysis. There are checkboxes to indicate the calibrations to be performed and a text box where the hyperparameter related to the basic configuration time can be modified.

2. AUDIO (Start: press `Start Audio Analysis`. End: click on the terminal and press `Enter` on keyboard): perform an audio analysis using live stream audio from a microphone (`16 bit` and `48000 Hz`).
   - VAD configuration: same as offline audio analysis
   - User configuration (Start: check `Calibrate user audio`, press `Start Audio Analysis`. End: click on the terminal and press `Enter` on keyboard): mode where the user has to speak using a neutral voice. This allow the system to calibrate the audio of that user.

# Hardware Feedback System

In order to represent feedback effectively, we decide to create a system capable managing and delivering various types of feedbacks to the user. More precisely, the system uses an Arduino to manage communications beteewn `feedback_execution.py`, interpreting the messages it sents and activating the right feedback in the specified way.

## Components and electrical scheme

The feedback system is composed by some main components:

- Arduino Uno: core of the system, responsible for controlling and communicating with all the devices connected to it.
- Vibration motor (4 devices): electric actuator which creates haptic feedback using vibrations.
- Peltier device (2 devices): it is an active solid-state heat pump that transfers heat from one side of the device to the other. We used it to provide temperature feedback, hot or cold, depending on which side of the device was in contact with the skin.

To connect devices to the system, we created prototype modular circuits, similar to each other. This was done in order to make them as replicable and easy to manage as possible.

![Electrical scheme of the circuits for vibration motor (left) and Peltier device (right).](/Arduino/electrical_scheme/Schematic_Research-Project_2024-03-13.png)

As shown in Figure \ref{fig:devices_scheme}, it is mentioned that both circuits have to be connected to a PWM pin. We used the open-source electronics platform Arduino UNO Rev3, so the PWM pins used are 3,5,6 and 9 for the vibration motors and 10 and 11 for the Peltier devices.

Moreover, the VCC of the Peltier device's circuit is referred to a maximum, as it is related to the characteristic of the Peltier device used. For example, to perform our tests, we used a 6V - 4A battery and 12V - 4A battery.

## Communication protocol

In order to map features into feedbacks, we created a little protocol to control vibration motors and Peltier devices. It consists in a simple string sent to the Arduino with the format `<code,value,pattern,min_intensity,max_intensity,pace>`:

- `code` [`h, v`]: it indicates the choice of control the Peltier cells `h` or the vibration motors (`v`).
- `value` [`0 - 5`]: it specifies which components control among Peltier devices (`<h,0,` first device and `<h,1` and second device) vibrations motors (`<v,0` first motor, ..., `<v,4` motors 0 and 1 and `<v,5` motors 2 and 3).
- `pattern` [`0, 1`]: it indicates how the components will reach the following levels:
  - `Linear` [`0`]: the proper component reachs the level $\Large\frac{max_inensity + min_intensity}{2}$ and then stop.
  - `Sinusoidal` [`1`]: the proper component continue to go back and forth between values `max_inensity` and `min_intensity`.
- `min_intensity` and `max_intensity` [`0 - 99`]: this value represents the possible intensities that can be requested from the devices connected to the Arduino.
- `pace` [`0 - 99`]: specifies the interval that is passed through each cycle of the Arduino in order to reach the levels indicated by `min_intensity` and `max_intensity`. If the value is `0` or greater than the difference between the actual level and the level to reach, the latter will be reached in one cycle and, in case of `sinusoidal` pattern, every cycle the level will be respectively `min_intensity` and `max_intensity` without transictions.

To give some examples of messages, with an initial level of 5, the following messages have the below effect:

- `<h,0,0,23,23,5>`: the first Peltier device will take 4 cycles (i.e., levels `10`, `15`, `20` and `23`) then it remains at that level, until a new message.
- `<v,2,1,20,40,10>`: the third vibration motor will take 4 cycles to reach `max_intensity` (i.e., levels `15`, `25`, `35` and `40`). Then it will have a fluctuating progress between `min_intensity` and `max_intensity` using 2 cycles each times.

# Acknowledgments

We express our gratitude to the developers of the tools and libraries used in this project, including the Social Signal Interpretation (SSI) framework, Mediapipe, and OpenSmile.
