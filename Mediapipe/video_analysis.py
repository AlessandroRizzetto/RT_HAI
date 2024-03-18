import json
import mediapipe as mp
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from numpy.linalg import norm
import time
import socket
import socket
import sys
import csv
import serial
import pandas as pd
import keyboard

COORDINATES = ['x', 'y', 'z']
HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 2222      # The port used by the server
SYNC = 1111
NETWORK_MESSAGE = 'SSI:STRT:RUN1\0'
start_time = None
elapsed_time = None
user_body = {}
arduino_is_connected = False
# ssi_is_connected = sys.argv[1] if len(sys.argv) > 1 else 0
# is_online = sys.argv[2] if len(sys.argv) > 2 else 1
# configuration_time = sys.argv[3] if len(sys.argv) > 3 else 5


# try: # to check if the arduino is connected to the PC and manage errors related to the serial communication
#     ser = serial.Serial('COM5', 9600, timeout=1)
#     ser.flush()
#     ser.reset_input_buffer()
#     arduino_is_connected = True
# except serial.SerialException as e:
#     print("The Arduino is not connected to the PC, using the test mode")
#     arduino_is_connected = False
#     pass




def manage_socket(host, port, ssi_is_connected, state):
    print("SSI is connected: ", ssi_is_connected)
    if state == "start":
        # ssi_is_connected = input("Is the SSI connected? Press 0 if it is not connected, 1 if it is connected")
        if ssi_is_connected == True:
            # create a socket and send data to the server
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client_socket.sendto(bytes(NETWORK_MESSAGE, "utf-8"), (HOST, SYNC))
            print("Socket connected")
            # client_socket.connect((host, port))  # connect to the server
            # print("Connected to server")
            return client_socket, ssi_is_connected
        elif ssi_is_connected == False:
            return None, ssi_is_connected
        else:
            print("Wrong input, please try again")
            manage_socket(host, port, ssi_is_connected, state)
    if state == "stop":
        if ssi_is_connected == "1":
            # client_socket.sendto(bytes("SSI:STOP:RUN1\0", "utf-8"), (HOST, SYNC)) # send a message to the server to stop the acquisition
            print("Socket closed")
            client_socket.close()
            return None, ssi_is_connected
        elif ssi_is_connected == "0":
            return None, ssi_is_connected


def serial_communication(message, LAST_MESSAGE, value, arduino_is_connected = arduino_is_connected):
    # --- ARDUINO-PC SERIAL COMMUNICATION SECTION --
    # COM4 is the port number of the Arduino
    # if arduino_is_connected:
    #     # print(LAST_MESSAGE, message)

    #     if message == "HANDS_NOT_VISIBILITY" and LAST_MESSAGE[0] == 0:
    #         ser.write(f"<v,4,1,30,60,10>\n".encode('utf-8')) # sinusoide
    #         print("Message sent to the arduino, hands not visible")
    #         LAST_MESSAGE[0] = 1
    #     if message == "HANDS_VISIBILITY" and LAST_MESSAGE[0] == 1:
    #         ser.write(f"<v,4,1,0,0,0>\n".encode('utf-8'))
    #         print("Message sent to the arduino, hands visible")
    #         LAST_MESSAGE[0] = 0

    #     if message == "HANDS_TOUCHING" and LAST_MESSAGE[1] == 0:
    #         ser.write(f"<v,4,0,40,40,0>\n".encode('utf-8'))
    #         print("Message sent to the arduino, hands touching")
    #         LAST_MESSAGE[1] = 1
    #     if message == "HANDS_NOT_TOUCHING" and LAST_MESSAGE[1] == 1:
    #         ser.write(f"<v,4,0,0,0,0>\n".encode('utf-8'))
    #         print("Message sent to the arduino, hands not touching")
    #         LAST_MESSAGE[1] = 0

    #     if message == "BAD_BODY_DIRECTION" and LAST_MESSAGE[2] == 0:
    #         if value == "Body Right":
    #             ser.write(f"<v,3,0,30,30,0>\n".encode('utf-8'))
    #             print("Message sent to the arduino, body right")
    #         elif value == "Body Left":
    #             ser.write(f"<v,2,0,30,30,0>\n".encode('utf-8'))
    #             print("Message sent to the arduino, body left")
    #         LAST_MESSAGE[2] = 1
    #     if message == "GOOD_BODY_DIRECTION" and LAST_MESSAGE[2] == 1:
    #         ser.write(f"<v,5,0,0,0,0>\n".encode('utf-8'))
    #         print("Message sent to the arduino, body forward")
    #         LAST_MESSAGE[2] = 0

    #     if message == "CROUCH" and LAST_MESSAGE[3] == 0:
    #         ser.write(f"<v,5,1,20,40,0>\n".encode('utf-8'))
    #         print("Message sent to the arduino, crouch")
    #         LAST_MESSAGE[3] = 1
    #     if message == "NOT_CROUCH" and LAST_MESSAGE[3] == 1:
    #         ser.write(f"<v,5,0,0,0,0>\n".encode('utf-8'))
    #         print("Message sent to the arduino, not crouch")
    #         LAST_MESSAGE[3] = 0

    #     if message == "BAD_HEAD_DIRECTION" and LAST_MESSAGE[4] == 0:
    #         ser.write(f"<h,0,0,40,40,5>\n".encode('utf-8'))
    #         print("Message sent to the arduino, head not forward")
    #         LAST_MESSAGE[4] = 1
    #     if message == "GOOD_HEAD_DIRECTION" and LAST_MESSAGE[4] == 1:
    #         ser.write(f"<h,0,0,0,0,5>\n".encode('utf-8'))
    #         print("Message sent to the arduino, head forward")
    #         LAST_MESSAGE[4] = 0
            
    return LAST_MESSAGE

def settings(start_time):
    time.sleep(0.5)
    video_is_over = False
    if start_time == None:
        start_time = time.time()
    
    return start_time, configuration_time, is_online, video_is_over
    

def OnlineOffline_management(is_online, configData): # function to manage the online and offline mode
    if not is_online:
        video_path =  configData["video_path"]
        cap = cv2.VideoCapture(video_path)
        print("Offline mode - total number of frames: ", cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    else:
        cap = cv2.VideoCapture(0)
    frame_counter = 0
    return cap, frame_counter

def offline_functions(client_socket, dataTable, LAST_MESSAGE, frame_counter, cap, configData): # TO DO: not working properly, video not stopping when it is over
    frame_counter += 1  
    print("frame number: ", frame_counter, " / ", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if frame_counter == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            video_is_over = True
            print("Video is over")
    else:
        video_is_over = False
    return video_is_over, frame_counter

def offline_overall_outcomes(client_socket, dataTable, LAST_MESSAGE, featuresTable, csv_file): # function to compute the overall features of the video
    dataframe = pd.read_csv(csv_file)
    most_frequent_features = {}
    # compute the mean of the features
    features_coloumns = ['crouch', 'hands_distance', 'hands_visibility', 'body_direction', 'head_direction'] # TO DO: add the other features, 'body_direction', 'head_direction'
    print(" \n Final Outcomes:")
    for feature in features_coloumns:
        values_count = dataframe[feature].value_counts()
        featuresTable[feature] = values_count.idxmax()
        most_frequent_features[feature] = values_count.idxmax()
        print(f"Most frequent {feature}: ", values_count.idxmax())
    

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
  
        

def bodyAndFace_inclination(client_socket, results, face_2d, face_3d, LAST_MESSAGE, image, img_h, img_w, img_c, dataTable, featuresTable, configData):
    body_2d = []
    body_3d = []
    text = ""
    
    if results.face_landmarks is not None:
        face_landmarks = results.face_landmarks
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in [33, 263, 1, 61, 291, 199]:
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    
                x, y = int(lm.x * img_w), int(lm.y * img_h)

                # Get the 2D Coordinates
                face_2d.append([x, y])
                # Get the 3D Coordinates
                face_3d.append([x, y, lm.z])
        
        for landmark in ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]:
            x, y = int(dataTable[f'{landmark}_x'][-1] * img_w), int(dataTable[f'{landmark}_y'][-1] * img_h)
            # Get the 2D Coordinates
            body_2d.append([x, y])
            # Get the 3D Coordinates
            body_3d.append([x, y, dataTable[f'{landmark}_z'][-1]])
        
        # Convert lists to NumPy arrays
        face_2d = np.array(face_2d, dtype=np.float64)
        body_2d = np.array(body_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        body_3d = np.array(body_3d, dtype=np.float64)
        
        # Camera matrix
        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                               [0, focal_length, img_w / 2],
                               [0, 0, 1]])

        # Distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        body_success, body_rot_vec, body_trans_vec = cv2.solvePnP(body_3d, body_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, _ = cv2.Rodrigues(rot_vec)
        body_rmat, _ = cv2.Rodrigues(body_rot_vec)

        # Get angles
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        body_angles, _, _, _, _, _ = cv2.RQDecomp3x3(body_rmat)

        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360
        body_x = body_angles[0] * 360
        body_y = body_angles[1] * 360
        body_z = body_angles[2] * 360

        # See where the user's head tilting
        if y < -10:
            text = "Looking Right"
        elif y > 10:
            text = "Looking Left"
        elif x < -10:
            text = "Looking Down"
        elif x > 10:
            text = "Looking Up"
        else:
            text = "Looking Forward"
        if body_y < - configData["body_y"]:
            body_text = "Body Right"
            LAST_MESSAGE = serial_communication("BAD_BODY_DIRECTION", LAST_MESSAGE, "Body Right")
        elif body_y > configData["body_y"]:
            body_text = "Body Left"
            LAST_MESSAGE = serial_communication("BAD_BODY_DIRECTION", LAST_MESSAGE, "Body Left")
        else:
            body_text = "Body Forward"
            LAST_MESSAGE = serial_communication("GOOD_BODY_DIRECTION", LAST_MESSAGE, 0)
        # if body_text != "Body Forward":
        #     print("Body direction: ", body_text)

        featuresTable[f'body_direction'].append(body_text)
        featuresTable[f'head_direction'].append(text)

        if len(featuresTable[f'head_direction']) > 30:
            features = featuresTable[f'head_direction'][-120:]
            values_count = pd.Series(features).value_counts()
            most_frequent_head_direction = values_count.idxmax()
            # print("Most frequent head direction: ", most_frequent_head_direction)
            if most_frequent_head_direction == "Looking Forward":
                # print("sent message to the arduino, good head direction")
                LAST_MESSAGE = serial_communication("GOOD_HEAD_DIRECTION", LAST_MESSAGE, 0)
            else:
                # print("sent message to the arduino, bad head direction")
                LAST_MESSAGE = serial_communication("BAD_HEAD_DIRECTION", LAST_MESSAGE, most_frequent_head_direction)

        # Display the nose and body direction
        nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
        body_3d_projection, _ = cv2.projectPoints(body_3d, body_rot_vec, body_trans_vec, cam_matrix, dist_matrix)

        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

        cv2.line(image, p1, p2, (255, 0, 0), 3)

        # Add text and lines to the image
        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # LAST_MESSAGE = serial_communication("HEAD", LAST_MESSAGE, y)
    return text
        
def body_settings(client_socket, dataTable, LAST_MESSAGE): # function to set the body of the user as the standard one
    user_body = dataTable.copy()
    # set values for the body as the mean of the values adding appending the mean to the list
    for name in user_body:
        user_body[name][-1] = np.mean(user_body[name])
        
    # calculate the standard bounding triangle (configuration of the body)
    standard_bounding_triangle, standard_y_torso = bounding_triangle(client_socket, user_body, LAST_MESSAGE)
    print("standard_bounding_triangle: ", standard_bounding_triangle)
    print("body_settings completed")
    return user_body, standard_bounding_triangle, standard_y_torso

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
    #print("bounding_triangle area: ", triangle_area)
    # calculate the y distance between the shoulders and the point between the hips
    y_torso = torso_points[0][1] - torso_points[3][1]
    return triangle_area, y_torso

def crouch_detection(client_socket, dataTable, LAST_MESSAGE, user_body, triangle_area, standard_bounding_triangle, standart_yTorso , yTorso, featuresTable, configData):
    bounding_area_proportion = configData["bounding_area_proportion"]
    yTorso_proportion = configData["yTorso_proportion"]
    #print("Bounding", standard_bounding_triangle, triangle_area)
    #print("torso", standart_yTorso, yTorso)
    if abs(yTorso) < abs(standart_yTorso * yTorso_proportion):
        crouch_is_good = False
        featuresTable[f'crouch'].append("Crouched") 
        # print("Crouch detected, yTorso is too small")
    # elif abs(triangle_area) < abs(standard_bounding_triangle * bounding_area_proportion):
    #     crouch_is_good = False
    #     featuresTable[f'crouch'].append("Crouched") 
    #     # print("Crouch detected, triangle area is too small")
    else:
        crouch_is_good = True
        featuresTable[f'crouch'].append("Not crouched")
    # if the user is crouching, send a message to the arduino to turn on the vibration
    if abs(yTorso) < abs(standart_yTorso * yTorso_proportion) or abs(triangle_area) < abs(standard_bounding_triangle * bounding_area_proportion):
        LAST_MESSAGE = serial_communication("CROUCH", LAST_MESSAGE, 0)
    else:
        LAST_MESSAGE = serial_communication("NOT_CROUCH", LAST_MESSAGE, 0)

def touching_hands(client_socket, dataTable, LAST_MESSAGE, featuresTable, configData):
    # calculate the distance between the hands
    hands_distance = abs(dataTable[f'LEFT_WRIST_x'][-1] - dataTable[f'RIGHT_WRIST_x'][-1]) + abs(dataTable[f'LEFT_WRIST_y'][-1] - dataTable[f'RIGHT_WRIST_y'][-1]) + abs(dataTable[f'LEFT_WRIST_z'][-1] - dataTable[f'RIGHT_WRIST_z'][-1])
    # print("hands_distance: ", hands_distance)
    hands_threshold = configData["hands_distance"] + 0.1
    if hands_distance < hands_threshold:
        LAST_MESSAGE = serial_communication("HANDS_TOUCHING", LAST_MESSAGE, 0)
        featuresTable[f'hands_distance'].append("Hands touching")
        # print("Hands touching")
        # serial_communication("HANDS_TOUCHING", LAST_MESSAGE, 0)
    else:
        LAST_MESSAGE = serial_communication("HANDS_NOT_TOUCHING", LAST_MESSAGE, 0)
        featuresTable[f'hands_distance'].append("Hands not touching")
        # print("Hands not touching")
        # serial_communication("HANDS_NOT_TOUCHING", LAST_MESSAGE, 0)
    return hands_distance

def hands_visibility(client_socket, dataTable, LAST_MESSAGE, featuresTable, configData):
    # understand if the hands are visible or not
    # print("LEFT_WRIST_visibility: ", dataTable[f'LEFT_WRIST_visibility'][-1])
    # print("RIGHT_WRIST_visibility: ", dataTable[f'RIGHT_WRIST_visibility'][-1])
    if dataTable[f'LEFT_WRIST_visibility'][-1] <= 0.4 and dataTable[f'RIGHT_WRIST_visibility'][-1] <= 0.4:
        hands_are_visible = False
        featuresTable[f'hands_visibility'].append("Hands not visible")
        #print("Hands not visible")
        LAST_MESSAGE = serial_communication("HANDS_NOT_VISIBILITY", LAST_MESSAGE, 0)
    else:
        hands_are_visible = True
        featuresTable[f'hands_visibility'].append("Hands visible")
        LAST_MESSAGE = serial_communication("HANDS_VISIBILITY", LAST_MESSAGE, 0)
        #print("Hands visible")
    
    
def compute_main_features(dataTable, name, COORDINATES, window_length, polyorder, LAST_MESSAGE): # function to compute the main features of the data (velocity, acceleration, kinetic energy)
    for coord in COORDINATES:
        data_col = dataTable[f'{name}_{coord}']
        data_col[-17:] = savgol_filter(data_col[-17:], window_length, polyorder)
        vel_col = f'{name}_{coord}_v'
        vel_value = abs(data_col[-1] - data_col[-17])
        dataTable[vel_col].append(vel_value)
    
    # compute acceleration as the derivative of the velocity
    acc_col = f'{name}_acceleration'
    acc_values = np.array([savgol_filter(dataTable[f'{name}_{coord}_v'][-17:], window_length, polyorder) for coord in COORDINATES])
    acc_values = np.diff(acc_values, 2, axis=0)
    acc_magnitude = norm(acc_values)
    acc_magnitude = round(acc_magnitude, 5)
    acc_magnitude = normalize(acc_magnitude)
    dataTable[acc_col].append(acc_magnitude)
    
    
    # compute kinetic energy as the square of the velocity
    vel_sum = sum([dataTable[f'{name}_{coord}_v'][-1] for coord in COORDINATES])
    dataTable[f'kinetic_Energy_{name}'].append(normalize((vel_sum**2)/2))
    

    
 
def mediaPipe(client_socket, ssi_is_connected, configData, is_online, configuration_time):
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
    start_time = None
    configuration_isdone = False
    start_time, configuration_time, is_online, video_is_over = settings(start_time)
    cap, frame_counter = OnlineOffline_management(is_online, configData)
    
    # Savgol filter parameters
    window_length = 17
    polyorder = 2
    LAST_MESSAGE = [0,0,0,0,0]

    landmarks_name = ['NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST',
                      'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP',
                      'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
                      'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
    # landmarks_name = ['NOSE']

    
    # create a dictionary with the landmarks as keys
    dataTable = {}
    featuresTable = {}
    # create a table with the landmarks as columns
    for name in landmarks_name:
        dataTable[f'{name}_x'] = []
        dataTable[f'{name}_y'] = []
        dataTable[f'{name}_z'] = []
        dataTable[f'{name}_x_v'] = []
        dataTable[f'{name}_y_v'] = []
        dataTable[f'{name}_z_v'] = []
        dataTable[f'{name}_visibility'] = []
        dataTable[f'kinetic_Energy_{name}'] = []
        dataTable[f'{name}_acceleration'] = []
    featuresTable[f'body_direction'] = []
    featuresTable[f'head_direction'] = []
    featuresTable[f'crouch'] = []
    featuresTable[f'hands_distance'] = []
    featuresTable[f'hands_visibility'] = []    
        
    
    
        
    with open("landmarks.stream~", "a+") as f:
        with holistic as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make Detections for the face 
                mp_face_mesh = mp.solutions.face_mesh
                face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
                
                # Make Detections
                results = holistic.process(image)
                # Faceresults = face_mesh.process(image)
                # Recolor image back to BGR for rendering
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Get the coordinates of the image
                img_h, img_w, img_c = image.shape
                face_3d = []
                face_2d = []
                # Pose Detections
                landmarks = results.pose_world_landmarks.landmark
                # Faceresults = results.face_landmarks
                
                if not is_online:
                    video_is_over, frame_counter = offline_functions(client_socket, dataTable, LAST_MESSAGE, frame_counter, cap, configData)

                for name in landmarks_name:
                    # append the coordinates of the landmark to the list
                    dataTable[f'{name}_x'].append(
                        landmarks[mp_holistic.PoseLandmark[name].value].x)
                    dataTable[f'{name}_y'].append(
                        landmarks[mp_holistic.PoseLandmark[name].value].y)
                    dataTable[f'{name}_z'].append(
                        landmarks[mp_holistic.PoseLandmark[name].value].z)
                    dataTable[f'{name}_visibility'].append(
                        landmarks[mp_holistic.PoseLandmark[name].value].visibility)
                    
                    
                    # calculate the velocities as the distance between the current and the previous frame
                    if len(dataTable[f'{name}_x']) > 17:                        
                        
                        compute_main_features(dataTable, name, COORDINATES, window_length, polyorder, LAST_MESSAGE)
                                            
                                                
                    else:
                        dataTable[f'{name}_x_v'].append(0) # append 0 if the list is empty
                        dataTable[f'{name}_y_v'].append(0) 
                        dataTable[f'{name}_z_v'].append(0)
                        dataTable[f'kinetic_Energy_{name}'].append(0)
                        dataTable[f'{name}_acceleration'].append(0)
                    if len(dataTable[f'{name}_x']) > 17:
                        # remove the first element of the list (the oldest one) to keep the list with the same length
                        dataTable[f'{name}_x'].pop(0)
                        dataTable[f'{name}_y'].pop(0)
                        dataTable[f'{name}_z'].pop(0)
                        dataTable[f'{name}_x_v'].pop(0)
                        dataTable[f'{name}_y_v'].pop(0)
                        dataTable[f'{name}_z_v'].pop(0)
                        dataTable[f'{name}_visibility'].pop(0)
                        dataTable[f'kinetic_Energy_{name}'].pop(0)
                        dataTable[f'{name}_acceleration'].pop(0)
                    
                    
                if len(dataTable[f'NOSE_x']) == 1:
                    # cancello il file csv se esiste
                    if os.path.exists('../Mediapipe/dataTable.csv'):
                        os.remove('../Mediapipe/dataTable.csv')
                
                
                    
                
                
                ##############################################################################
                
                elapsed_time = time.time() - start_time
                if elapsed_time > configuration_time and configuration_isdone == False:
                    user_body, standard_bounding_triangle, standart_yTorso = body_settings(client_socket, dataTable, LAST_MESSAGE)
                    configuration_isdone = True
                
                ##############################################################################
                # ESECUZIONE EFFETTIVA DELLE FUNZIONI
                if configuration_isdone == True:
                    #test nose position threshold
                    #nose_test(client_socket, dataTable, LAST_MESSAGE)
                    #print(dataTable[f'NOSE_acceleration'][-1])
                    triangle_area, yTorso = bounding_triangle(client_socket, dataTable, LAST_MESSAGE)
                    #print(dataTable[f'NOSE_x'][-1], dataTable[f'NOSE_y'][-1], dataTable[f'NOSE_z'][-1])
                    crouch_detection(client_socket, dataTable, LAST_MESSAGE, user_body, triangle_area, standard_bounding_triangle, standart_yTorso, yTorso,featuresTable, configData)
                    touching_hands(client_socket, dataTable, LAST_MESSAGE, featuresTable, configData)
                    hands_visibility(client_socket, dataTable, LAST_MESSAGE, featuresTable, configData)
                    # call a function that do the head inclination detection
                    headAlert = bodyAndFace_inclination(client_socket, results, face_2d, face_3d, LAST_MESSAGE, image, img_h, img_w, img_c, dataTable, featuresTable, configData)
                    # create and update the csv file with the data
                    
                ##################################################################################
                if len(featuresTable[f'crouch']) > 1:
                        # se non esiste creo un file csv con i nomi delle colonne
                        if not os.path.exists('../Mediapipe/dataTable.csv'):
                            with open('../Mediapipe/dataTable.csv', 'a+') as f:
                                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                                writer.writerow(['class'] + [f'{name}_x' for name in landmarks_name] + [f'{name}_y' for name in landmarks_name] + [f'{name}_z' for name in landmarks_name] + [f'{name}_x_v' for name in landmarks_name] + [f'{name}_y_v' for name in landmarks_name] + [f'{name}_z_v' for name in landmarks_name] 
                                                + [f'kinetic_Energy_{name}' for name in landmarks_name] + [f'{name}_acceleration' for name in landmarks_name] + [f'{name}_visibility' for name in landmarks_name] + ['crouch', 'hands_distance', 'hands_visibility'
                                                                                                                                                                                                                    , 'body_direction', 'head_direction' 
                                                                                                                                                                                                                      ] )
                        #print(featuresTable)
                        ai_data = list(np.array( [dataTable[f'{name}_x'][-1] for name in landmarks_name] + [dataTable[f'{name}_y'][-1] for name in landmarks_name] + [dataTable[f'{name}_z'][-1] for name in landmarks_name] + [dataTable[f'{name}_x_v'][-1] for name in landmarks_name] + [dataTable[f'{name}_y_v'][-1] for name in landmarks_name] 
                                                + [dataTable[f'{name}_z_v'][-1] for name in landmarks_name] + [dataTable[f'kinetic_Energy_{name}'][-1] for name in landmarks_name] + [dataTable[f'{name}_acceleration'][-1] for name in landmarks_name] 
                                                + [dataTable[f'{name}_visibility'][-1] for name in landmarks_name] + [featuresTable[f'crouch'][-1], featuresTable[f'hands_distance'][-1], featuresTable[f'hands_visibility'][-1]
                                                , featuresTable[f'body_direction'][-1], featuresTable[f'head_direction'][-1]                                                                   
                                                                                                                      ] ))
                        ai_data.insert(0, "CLASS") # to change with the class of the user !!!
                        with open('../Mediapipe/dataTable.csv', 'a') as f:
                            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                            writer.writerow(ai_data)
               
                if is_online:
                    # Render landmarks
                    #1. Draw face landmarks
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

                if cv2.waitKey(10) & 0xFF == ord('q') or video_is_over or keyboard.is_pressed('enter'):
                    try:
                        offline_overall_outcomes(client_socket, dataTable, LAST_MESSAGE, featuresTable, "../Mediapipe/dataTable.csv")
                    except:
                        print("ERROR: dataTable.csv not found")
                    break
                

    cap.release()
    cv2.destroyAllWindows()
    
    # offline_overall_outcomes(client_socket, dataTable, LAST_MESSAGE, featuresTable, "dataTable.csv")
    manage_socket(HOST, PORT, ssi_is_connected, "stop")


if __name__ == "__main__":
    # Define the host and the port of the server
    host = "localhost"
    port = 9000
    
    if os.path.exists('scripts/body_config.json'):
        with open('scripts/body_config.json') as f:
            configData = json.load(f)
    elif os.path.exists('../Mediapipe/scripts/body_config.json'):
        with open('../Mediapipe/scripts/body_config.json') as f:
            configData = json.load(f)
        
    # check if the input parameters are correct
    if len(sys.argv) < 3:
        print("Usage: python video_analysis.py <ssi_is_connected> <is_online>")
        sys.exit(1)

    ssi_is_connected = sys.argv[1].lower() == 'true' 
    is_online = sys.argv[2].lower() == 'true'
    configuration_time = configData["configuration_time"] 
    # create a socket
    client_socket, ssi_is_connected = manage_socket(host, port, ssi_is_connected, state="start")
    #client_socket = 1
    # start mediapipe
    mediaPipe(client_socket, ssi_is_connected, configData, is_online, configuration_time)
  
