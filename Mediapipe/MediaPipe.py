import mediapipe as mp
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import time
import socket
import socket
import sys
import serial

COORDINATES = ['x', 'y', 'z']
HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 2222      # The port used by the server
SYNC = 1111
NETWORK_MESSAGE = 'SSI:STRT:RUN1\0'
start_time = None
elapsed_time = None
user_body = {}
# ser = serial.Serial('COM4', 9600, timeout=1)
# ser.flush()
# ser.reset_input_buffer()


def create_socket(host, port):
    # create a socket and send data to the server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.sendto(bytes(NETWORK_MESSAGE, "utf-8"), (HOST, SYNC))
    print("Socket connected")
    # client_socket.connect((host, port))  # connect to the server
    # print("Connected to server")
    return client_socket


def serial_communication(message, LAST_MESSAGE):
    # --- ARDUINO-PC SERIAL COMMUNICATION SECTION --
    # COM4 is the port number of the Arduino

    # print(LAST_MESSAGE, message)
    if message == 1 and LAST_MESSAGE == False:
        ser.write("VIBRATION_ON\n".encode('utf-8'))
        print("Vibration has been turned ON!")

        LAST_MESSAGE = True
    elif message == 0 and LAST_MESSAGE == True:
        ser.write("VIBRATION_OFF\n".encode('utf-8'))

        print("Vibration has been turned OFF!")
        LAST_MESSAGE = False

    return LAST_MESSAGE


def send_data_network(client_socket, data):
    # send data to the server
    client_socket.sendto(bytes(data + "\0", "utf-8"), (HOST, PORT))
    # receive data from the server
    # answer = client_socket.recv(1024)
    # return answer.decode()


def normalize(value): # to adapt when we will have the real values from experiment
    # normalize the data as a value from 0 to 1
    max_value = 1
    min_value = 0
    return (value - min_value) / (max_value - min_value)

def nose_test(client_socket, dataTable, LAST_MESSAGE):
    # test 
    nose_test = 1 - dataTable[f'{"NOSE"}_y'][-1]
    # send data to the server
    send_data_network(client_socket, str(
        nose_test) + "\n")

    if dataTable[f'{"NOSE"}_y'][-1] < 0.5:
        LAST_MESSAGE = serial_communication(1, LAST_MESSAGE)
    else:
        LAST_MESSAGE = serial_communication(0, LAST_MESSAGE)

def body_settings(client_socket, dataTable, LAST_MESSAGE):
    
    user_body = dataTable.copy()
    # set values for the body as the mean of the values adding appending the mean to the list
    for name in user_body:
        user_body[name][-1] = np.mean(user_body[name])
        
     
   
    standard_bounding_triangle = bounding_triangle(client_socket, user_body, LAST_MESSAGE)
    print("standard_bounding_triangle: ", standard_bounding_triangle)
    print("body_settings completed")
    return user_body, standard_bounding_triangle

def bounding_triangle(client_socket, dataTable, LAST_MESSAGE):
    # calculate the bounding triangle thanks to the coordinates of the landmarks
    # calculate the area of the triangle
    torso_points = []
    for landmark in ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]:
        point = [dataTable[f'{landmark}_{coord}'][-1] for coord in COORDINATES]
        torso_points.append(point)
    # calculate the point beetwen the hips as the mean of the coordinates of the two hips and concatenate it to the list
    point = [(torso_points[2][i] + torso_points[3][i])/2 for i in range(3)]
    torso_points.append(point)
    torso_points = np.array(torso_points)
    # calculate tridimensional area of the triangle between the shoulders and the point between the hips
    triangle_area = 1-(np.linalg.norm(np.cross(torso_points[0]-torso_points[1], torso_points[0]-torso_points[3]))/2)
    print("bounding_triangle area: ", triangle_area)
    return triangle_area

def crouch_detection(client_socket, dataTable, LAST_MESSAGE, user_body, triangle_area, standard_bounding_triangle):
    bounding_area_threshold = 0.2
    if triangle_area < standard_bounding_triangle - bounding_area_threshold:
        print("Crouch detected")
    
    

def mediaPipe(client_socket):
    global configuration
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
    configuration_isdone = False
    # Savgol filter parameters
    window_length = 17
    polyorder = 2
    LAST_MESSAGE = False

    landmarks_name = ['NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST',
                      'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP',
                      'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
                      'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
    # landmarks_name = ['NOSE']

    # Initialize the DataFrame
    data = pd.DataFrame(columns=[f'{landmark}_x' for landmark in landmarks_name] +
                        [f'{landmark}_y' for landmark in landmarks_name] +
                        [f'{landmark}_x_v' for landmark in landmarks_name] +
                        [f'{landmark}_y_v' for landmark in landmarks_name] +
                        [f'kinetic_Energy_{landmark}' for landmark in landmarks_name])
    # Initialize at 0 the first velocities
    data.loc[0, [f'{landmark}_x_v' for landmark in landmarks_name]] = 0
    data.loc[0, [f'{landmark}_y_v' for landmark in landmarks_name]] = 0
    data.loc[0, [
        f'kinetic_Energy_{landmark}' for landmark in landmarks_name]] = 0
    data.loc[0, [f'{landmark}_x' for landmark in landmarks_name]] = 0
    data.loc[0, [f'{landmark}_y' for landmark in landmarks_name]] = 0
    # print(data)

    dataTable = {}
    # create a table with the landmarks as columns
    for name in landmarks_name:
        dataTable[f'{name}_x'] = []
        dataTable[f'{name}_y'] = []
        dataTable[f'{name}_z'] = []
        dataTable[f'{name}_x_v'] = []
        dataTable[f'{name}_y_v'] = []
        dataTable[f'{name}_z_v'] = []
        dataTable[f'kinetic_Energy_{name}'] = []
        # initialize the first velocities at 0
        # dataTable[f'{name}_x_v'].append(0)
        # dataTable[f'{name}_y_v'].append(0)
        # dataTable[f'kinetic_Energy_{name}'].append(0)
        # dataTable[f'{name}_x'].append(0)
        # dataTable[f'{name}_y'].append(0)

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
                
               
                
                # landmarks_coords = []

                for name in landmarks_name:
                    # append the coordinates of the landmark to the list
                    # landmarks_coords.append([landmarks[mp_holistic.PoseLandmark[name].value].x,
                    #                         landmarks[mp_holistic.PoseLandmark[name].value].y])
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

                    dataTable[f'{name}_x'].append(
                        landmarks[mp_holistic.PoseLandmark[name].value].x)
                    dataTable[f'{name}_y'].append(
                        landmarks[mp_holistic.PoseLandmark[name].value].y)
                    dataTable[f'{name}_z'].append(
                        landmarks[mp_holistic.PoseLandmark[name].value].z)
                    # Structure version
                    # calculate the velocities as the distance between the current and the previous frame
                    if len(dataTable[f'{name}_x']) > 17:
                        # apply the savgol filter to the last 17 values
                        # dataTable[f'{name}_x'][-17:] = savgol_filter(
                        #     dataTable[f'{name}_x'][-17:], window_length, polyorder)
                        # dataTable[f'{name}_y'][-17:] = savgol_filter(
                        #     dataTable[f'{name}_y'][-17:], window_length, polyorder)
                        # dataTable[f'{name}_z'][-17:] = savgol_filter(
                        #     dataTable[f'{name}_z'][-17:], window_length, polyorder)
                        # # calculate the velocities as the distance between the current and the previous frame
                        # dataTable[f'{name}_x_v'].append(
                        #     abs(dataTable[f'{name}_x'][-1] - dataTable[f'{name}_x'][-17]))
                        # dataTable[f'{name}_y_v'].append(
                        #     abs(dataTable[f'{name}_y'][-1] - dataTable[f'{name}_y'][-17]))
                        # dataTable[f'{name}_z_v'].append(
                        #     abs(dataTable[f'{name}_z'][-1] - dataTable[f'{name}_z'][-17]))
                        
                        # stessa cosa di sopra ma scritta meglio
                        for coord in COORDINATES:
                            data_col = dataTable[f'{name}_{coord}']
                            data_col[-17:] = savgol_filter(data_col[-17:], window_length, polyorder)
                            
                            vel_col = f'{name}_{coord}_v'
                            vel_value = abs(data_col[-1] - data_col[-17])
                            dataTable[vel_col].append(vel_value)

                        # calculate the kinetic energy as half of the sum of the square of the velocities, normalized from 0 to 1
                        # dataTable[f'kinetic_Energy_{name}'].append(
                        #     normalize(dataTable[f'{name}_x_v'][-1]**2 + dataTable[f'{name}_y_v'][-1]**2)/2)
                        dataTable[f'kinetic_Energy_{name}'].append(
                            normalize(((dataTable[f'{name}_x_v'][-1] + dataTable[f'{name}_y_v'][-1] + dataTable[f'{name}_z_v'][-1])**2)/2))

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
                    # print(dataTable[f'kinetic_Energy_{name}'][-1])
                    # print(dataTable[f'{name}_x'])
                    # print(name)
                
                
                # Pandas Version
                # new_data = {}
                # for idx, name in enumerate(landmarks_name):
                #     # idx is the index of the landmark in the list
                #     new_data[f'{name}_x'] = landmarks_coords[idx][0]
                #     new_data[f'{name}_y'] = landmarks_coords[idx][1]

                # # add new data to the dataframe as a new row
                # data = data._append(new_data, ignore_index=True)

                # if len(data) > 17:
                #     for idx, name in enumerate(landmarks_name):
                #         data.at[data.index[-1], f'{name}_x_v'] = abs(
                #             data.at[data.index[-1], f'{name}_x'] - data.at[data.index[-17], f'{name}_x'])
                #         data.at[data.index[-1], f'{name}_y_v'] = abs(
                #             data.at[data.index[-1], f'{name}_y'] - data.at[data.index[-17], f'{name}_y'])
                #         data.at[data.index[-1], f'kinetic_Energy_{name}'] = data.at[data.index[-1],
                #                                                                     f'{name}_x_v']**2 + data.at[data.index[-1], f'{name}_y_v']**2
                # if len(data) > 17:
                #     # remove the first element of the list (the oldest one) to keep the list with the same length
                #     data = data.drop(data.index[0])

                # # print(data)
                # data.to_csv('data.csv', index=False)

                # # save data into .stream file
                # f.write(str(1-nose[1]) + "\n")
                # f.flush()
                # os.fsync(f.fileno())  # flush the buffer
                
                ##############################################################################
                # configuration of the body of the user
                if start_time == None:
                    start_time = time.time()
                elapsed_time = time.time() - start_time
                if elapsed_time > 5 and configuration_isdone == False:
                    user_body, standard_bounding_triangle = body_settings(client_socket, dataTable, LAST_MESSAGE)
                    configuration_isdone = True
                
                ##############################################################################
                # ESECUZIONE EFFETTIVA DELLE FUNZIONI
                if configuration_isdone == True:
                    #test nose position threshold
                    #nose_test(client_socket, dataTable, LAST_MESSAGE)
                    triangle_area = bounding_triangle(client_socket, dataTable, LAST_MESSAGE)
                    #print(dataTable[f'NOSE_x'][-1], dataTable[f'NOSE_y'][-1], dataTable[f'NOSE_z'][-1])
                    crouch_detection(client_socket, dataTable, LAST_MESSAGE, user_body, triangle_area, standard_bounding_triangle)
                ##################################################################################

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
    # client_socket.close()


if __name__ == "__main__":
    # Define the host and the port of the server
    host = "localhost"
    port = 9000

    # create a socket
    #client_socket = create_socket(host, port)
    client_socket = 1
    # start mediapipe
    mediaPipe(client_socket)
  
