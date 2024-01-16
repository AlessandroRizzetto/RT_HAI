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

To install the necessary dependencies for this project, run the following command:

```bash
./install.cmd
```

To install SSI, follow the instructions provided in the [official documentation](https://rawgit.com/hcmlab/ssi/master/docs/index.html#installation)

If you want to test the software only with the body movements, you can skip the installation of OpenSmile and the SSI framework.

## Usage

The software has two different modes of operation: real-time and offline. The former is used to analyze user behavior during interactions with the virtual agent, while the latter is employed for offline analysis of pre-recorded data.

To run the real-time analysis, execute the following command:

```bash
# If you want to analyze only the body movements
python3 Mediapipe\MediaPipe.py
# Then uncomment the prints that you want to see in the terminal
```

To run the offline analysis, execute the following command:

```bash
# If you want to analyze only the body movements
python3 Mediapipe\MediaPipe.py

# Make sure to change the path to the video file in the code at line 120
```

At the end of the analysis, the software will generate a CSV file containing the extracted data and prints a overall summary of the features.

# Hardware Feedback System

![Image Description](/Arduino/electrical_scheme/Scheme.JPG)

[Provide details about the hardware feedback system and its integration]

# Acknowledgments

We express our gratitude to the developers of the tools and libraries used in this project, including the Social Signal Interpretation (SSI) framework, Mediapipe, and OpenSmile.
