import mediapipe as mp
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# Setup MediaPipe Holistic instance
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Setup drawing utility
mp_drawing = mp.solutions.drawing_utils

# If exists previous data, delete it
if os.path.exists("landmarks.stream~"):
    os.remove("landmarks.stream~")
    print("landmarks.stream file removed")

# Make Detections
cap = cv2.VideoCapture(0)
start_time = None
# Savgol filter parameters
window_length = 17
polyorder = 2


# landmarks_name = ['NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST',
#                   'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP',
#                   'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
#                   'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
landmarks_name = ['NOSE']

# Initialize the DataFrame
data = pd.DataFrame(columns=[f'{landmark}_x' for landmark in landmarks_name] +
                    [f'{landmark}_y' for landmark in landmarks_name] +
                    [f'{landmark}_x_v' for landmark in landmarks_name] +
                    [f'{landmark}_y_v' for landmark in landmarks_name] +
                    [f'kinetic_Energy_{landmark}' for landmark in landmarks_name])
# Initialize at 0 the first velocities
data.loc[0, [f'{landmark}_x_v' for landmark in landmarks_name]] = 0
data.loc[0, [f'{landmark}_y_v' for landmark in landmarks_name]] = 0
data.loc[0, [f'kinetic_Energy_{landmark}' for landmark in landmarks_name]] = 0
data.loc[0, [f'{landmark}_x' for landmark in landmarks_name]] = 0
data.loc[0, [f'{landmark}_y' for landmark in landmarks_name]] = 0
# print(data)

dataTable = {}
# create a table with the landmarks as columns
for name in landmarks_name:
    dataTable[f'{name}_x'] = []
    dataTable[f'{name}_y'] = []
    dataTable[f'{name}_x_v'] = []
    dataTable[f'{name}_y_v'] = []
    dataTable[f'kinetic_Energy_{name}'] = []
    # initialize the first velocities at 0
    dataTable[f'{name}_x_v'].append(0)
    dataTable[f'{name}_y_v'].append(0)
    dataTable[f'kinetic_Energy_{name}'].append(0)
    dataTable[f'{name}_x'].append(0)
    dataTable[f'{name}_y'].append(0)

with open("landmarks.stream~", "a+") as f:
    with holistic as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make Detections
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract elbow landmarks
            # try:
            landmarks = results.pose_landmarks.landmark

            landmarks_coords = []

            for name in landmarks_name:
                # append the coordinates of the landmark to the list
                landmarks_coords.append([landmarks[mp_holistic.PoseLandmark[name].value].x,
                                         landmarks[mp_holistic.PoseLandmark[name].value].y])
                # data[f'{name}_x'] = landmarks[mp_holistic.PoseLandmark[name].value].x
                # data[f'{name}_y'] = landmarks[mp_holistic.PoseLandmark[name].value].y
                # data[f'{name}_x_v'] = 0
                # data[f'{name}_y_v'] = 0
                # data[f'kinetic_Energy_{name}'] = 0
                # calculate the velocities as the distance between the current and the previous frame
                # if len(data) > 1:
                #     data.at[data.index[-1], f'{name}_x_v'] = abs(
                #         data.at[data.index[-1], f'{name}_x'] - data.at[data.index[-2], f'{name}_x'])
                #     data.at[data.index[-1], f'{name}_y_v'] = abs(
                #         data.at[data.index[-1], f'{name}_y'] - data.at[data.index[-2], f'{name}_y'])
                # else:
                #     data.at[data.index[-1], f'{name}_x_v'] = 0
                #     data.at[data.index[-1], f'{name}_y_v'] = 0
                # print(data)

                # normalizzo le coordinate dei landmarks

                dataTable[f'{name}_x'].append(
                    landmarks[mp_holistic.PoseLandmark[name].value].x)
                dataTable[f'{name}_y'].append(
                    landmarks[mp_holistic.PoseLandmark[name].value].y)

            # Structure version
            # calculate the velocities as the distance between the current and the previous frame
            if len(dataTable[f'{name}_x']) > 17:
                dataTable[f'{name}_x_v'].append(
                    abs(dataTable[f'{name}_x'][-1] - dataTable[f'{name}_x'][-17]))
                dataTable[f'{name}_y_v'].append(
                    abs(dataTable[f'{name}_y'][-1] - dataTable[f'{name}_y'][-17]))
                dataTable[f'kinetic_Energy_{name}'].append(
                    dataTable[f'{name}_x_v'][-1]**2 + dataTable[f'{name}_y_v'][-1]**2)
            else:
                dataTable[f'{name}_x_v'].append(0)
                dataTable[f'{name}_y_v'].append(0)
                dataTable[f'kinetic_Energy_{name}'].append(0)
            if len(dataTable[f'{name}_x']) > 17:
                # remove the first element of the list (the oldest one) to keep the list with the same length
                dataTable[f'{name}_x'].pop(0)
                dataTable[f'{name}_y'].pop(0)
                dataTable[f'{name}_x_v'].pop(0)
                dataTable[f'{name}_y_v'].pop(0)
                dataTable[f'kinetic_Energy_{name}'].pop(0)
            print(dataTable[f'{name}_x_v'][-1])

            # Pandas Version
            new_data = {}
            for idx, name in enumerate(landmarks_name):
                # idx is the index of the landmark in the list
                new_data[f'{name}_x'] = landmarks_coords[idx][0]
                new_data[f'{name}_y'] = landmarks_coords[idx][1]

            # add new data to the dataframe as a new row
            data = data._append(new_data, ignore_index=True)

            if len(data) > 17:
                for idx, name in enumerate(landmarks_name):
                    data.at[data.index[-1], f'{name}_x_v'] = abs(
                        data.at[data.index[-1], f'{name}_x'] - data.at[data.index[-17], f'{name}_x'])
                    data.at[data.index[-1], f'{name}_y_v'] = abs(
                        data.at[data.index[-1], f'{name}_y'] - data.at[data.index[-17], f'{name}_y'])
                    data.at[data.index[-1], f'kinetic_Energy_{name}'] = data.at[data.index[-1],
                                                                                f'{name}_x_v']**2 + data.at[data.index[-1], f'{name}_y_v']**2
            if len(data) > 17:
                # remove the first element of the list (the oldest one) to keep the list with the same length
                data = data.drop(data.index[0])

            # print(data)
            data.to_csv('data.csv', index=False)

            # # save data into .stream file
            # f.write(str(1-nose[1]) + "\n")
            # f.flush()
            # os.fsync(f.fileno())  # flush the buffer

            # Render landmarks
            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(
                                          color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(
                                          color=(80, 256, 121), thickness=1, circle_radius=1)
                                      )

            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(
                                          color=(80, 44, 121), thickness=2, circle_radius=2)
                                      )

            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(
                                          color=(121, 44, 250), thickness=2, circle_radius=2)
                                      )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            cv2.imshow('MediaPipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
