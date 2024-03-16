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
import json

COORDINATES = ['x', 'y', 'z']

start_time = None
elapsed_time = None
user_body = {}

def settings(start_time):
    time.sleep(0.5)
    video_is_over = False
    if start_time == None:
        start_time = time.time()
    return start_time, configuration_time, video_is_over


def OnlineOffline_management(is_online): # function to manage the online and offline mode
    if not is_online:
        video_path = "C:/Users/Alessandro/Desktop/Research Project/RT_HAI/Videos/VideoProva_short.mp4" # to change with the path of the video you want to use
        cap = cv2.VideoCapture(video_path)
        print("Offline mode - total number of frames: ", cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    else:
        cap = cv2.VideoCapture(0)
    frame_counter = 0
    return cap, frame_counter

    
def saveCalibrationData(dataTable, featuresTable, csv_file, calibrationData, hands_calibration,crouch_calibration, body_direction_calibration, crouched_yTorso_custom_proportion, crouched_bound_triangle_custom_proportion):
    # save the calibration data in a json file
    dataframe = pd.read_csv(csv_file)
    most_frequent_features = {}
    # compute the mean of the features
    features_coloumns = ['crouch', 'hands_distance', 'hands_visibility', 'body_direction', 'head_direction'] 
    if hands_calibration:
        touching_hands_distance_mean = dataframe['hands_physical_distance'].mean()
        calibrationData["hands_distance"] = touching_hands_distance_mean
        print("touching_hands_distance_mean: ", touching_hands_distance_mean)
        
    if crouch_calibration:
        calibrationData["yTorso_proportion"] = dataframe['crouched_yTorso_custom_proportion'].mean()
        calibrationData["bounding_area_proportion"] = dataframe['crouched_bound_triangle_custom_proportion'].mean()
        print("crouched_yTorso_custom_proportion: ", calibrationData["yTorso_proportion"], "crouched_bound_triangle_custom_proportion: ", calibrationData["bounding_area_proportion"])
    
    if body_direction_calibration:
        body_y_mean = dataframe['body_y'].mean()
        calibrationData["body_y"] = body_y_mean
        print("body_y_mean: ", body_y_mean)
    
    with open('bodyCalibrations.json', 'w') as f:
            json.dump(calibrationData, f, indent=4)
        

        

def bodyAndFace_inclination( results, face_2d, face_3d, image, img_h, img_w, img_c, dataTable, featuresTable, calibrationData):
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
        if body_y < - calibrationData["body_y"]:
            body_text = "Body Right"
        elif body_y > calibrationData["body_y"]:
            body_text = "Body Left"
        else:
            body_text = "Body Forward"

        featuresTable[f'body_direction'].append(body_text)
        featuresTable[f'head_direction'].append(text)
        featuresTable[f'body_y'].append(body_y)

        

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

    return text
        
def body_settings(dataTable): # function to set the body of the user as the standard one
    user_body = dataTable.copy()
    # set values for the body as the mean of the values adding appending the mean to the list
    for name in user_body:
        user_body[name][-1] = np.mean(user_body[name])
        
    # calculate the standard bounding triangle (configuration of the body)
    standard_bounding_triangle, standard_y_torso = bounding_triangle(user_body)
    print("body_settings completed")
    return user_body, standard_bounding_triangle, standard_y_torso

def bounding_triangle(dataTable):
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
    # calculate the y distance between the shoulders and the point between the hips
    y_torso = torso_points[0][1] - torso_points[3][1]
    return triangle_area, y_torso

def crouch_detection(dataTable, user_body, triangle_area, standard_bounding_triangle, standart_yTorso , yTorso, featuresTable):
    bounding_area_proportion = 0.98 
    yTorso_proportion = 0.99
    if abs(yTorso) < abs(standart_yTorso * yTorso_proportion):
        crouch_is_good = False
        featuresTable[f'crouch'].append("Crouched") 
    elif abs(triangle_area) < abs(standard_bounding_triangle * bounding_area_proportion):
        crouch_is_good = False
        featuresTable[f'crouch'].append("Crouched") 
    else:
        crouch_is_good = True
        featuresTable[f'crouch'].append("Not crouched")
    
    # calculate the custom proportions of the crouched body with the standard one
    crouched_yTorso_custom_proportion = yTorso/standart_yTorso
    crouched_bound_triangle_custom_proportion = triangle_area/standard_bounding_triangle
    featuresTable[f'crouched_yTorso_custom_proportion'].append(crouched_yTorso_custom_proportion)
    featuresTable[f'crouched_bound_triangle_custom_proportion'].append(crouched_bound_triangle_custom_proportion)
    
    return crouched_yTorso_custom_proportion, crouched_bound_triangle_custom_proportion

def touching_hands(dataTable, featuresTable, calibrationData):
    # calculate the distance between the hands
    hands_distance = abs(dataTable[f'LEFT_WRIST_x'][-1] - dataTable[f'RIGHT_WRIST_x'][-1]) + abs(dataTable[f'LEFT_WRIST_y'][-1] - dataTable[f'RIGHT_WRIST_y'][-1]) + abs(dataTable[f'LEFT_WRIST_z'][-1] - dataTable[f'RIGHT_WRIST_z'][-1])
    featuresTable[f'hands_physical_distance'].append(hands_distance)
    if hands_distance < calibrationData["hands_distance"]:
        featuresTable[f'hands_distance'].append("Hands touching")
    else:
        featuresTable[f'hands_distance'].append("Hands not touching")
    return hands_distance

def hands_visibility(dataTable, featuresTable):
    # understand if the hands are visible or not
    if dataTable[f'LEFT_WRIST_visibility'][-1] <= 0.4 and dataTable[f'RIGHT_WRIST_visibility'][-1] <= 0.4:
        hands_are_visible = False
        featuresTable[f'hands_visibility'].append("Hands not visible")
    else:
        hands_are_visible = True
        featuresTable[f'hands_visibility'].append("Hands visible")
 
def mediaPipe(crouch_calibration, hands_calibration, body_direction_calibration, configuration_time, calibrationData):
    # Setup MediaPipe Holistic instance
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Setup drawing utility
    mp_drawing = mp.solutions.drawing_utils

    # Make Detections
    start_time = None
    configuration_isdone = False
    
    is_online = True
    start_time, configuration_time, video_is_over = settings(start_time)
    cap, frame_counter = OnlineOffline_management(is_online)
    
    # Savgol filter parameters
    window_length = 17
    polyorder = 2

    landmarks_name = ['NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST',
                      'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP',
                      'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
                      'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

    # create a dictionary with the landmarks as keys
    dataTable = {}
    featuresTable = {}
    # create a table with the landmarks as columns
    for name in landmarks_name:
        dataTable[f'{name}_x'] = []
        dataTable[f'{name}_y'] = []
        dataTable[f'{name}_z'] = []
        dataTable[f'{name}_visibility'] = []
    featuresTable[f'body_direction'] = []
    featuresTable[f'head_direction'] = []
    featuresTable[f'crouch'] = []
    featuresTable[f'hands_distance'] = []
    featuresTable[f'hands_visibility'] = []
    featuresTable[f'hands_physical_distance'] = []
    featuresTable[f'body_y'] = []
    featuresTable[f'crouched_bound_triangle_custom_proportion'] = []
    featuresTable[f'crouched_yTorso_custom_proportion'] = []
        
    
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
                
            if len(dataTable[f'{name}_x']) > 17:
                        # remove the first element of the list (the oldest one) to keep the list with the same length
                        dataTable[f'{name}_x'].pop(0)
                        dataTable[f'{name}_y'].pop(0)
                        dataTable[f'{name}_z'].pop(0)
                        dataTable[f'{name}_visibility'].pop(0)      
            if len(dataTable[f'NOSE_x']) == 1:
                    # cancello il file csv se esiste
                    if os.path.exists('configurationTable.csv'):
                        os.remove('configurationTable.csv')
                
            
            elapsed_time = time.time() - start_time
            if elapsed_time > configuration_time and configuration_isdone == False:
                user_body, standard_bounding_triangle, standart_yTorso = body_settings(dataTable)
                configuration_isdone = True
            
            # Call the functions to detect all the features
            if configuration_isdone == True:
                triangle_area, yTorso = bounding_triangle(dataTable)
                crouched_yTorso_custom_proportion, crouched_bound_triangle_custom_proportion = crouch_detection(dataTable, user_body, triangle_area, standard_bounding_triangle, standart_yTorso, yTorso,featuresTable)
                hands_distance = touching_hands( dataTable, featuresTable, calibrationData)
                hands_visibility( dataTable, featuresTable)
                # call a function that do the head inclination detection
                headAlert = bodyAndFace_inclination( results, face_2d, face_3d, image, img_h, img_w, img_c, dataTable, featuresTable, calibrationData)
                # create and update the csv file with the data
                
            if len(featuresTable[f'crouch']) > 1:
                    # se non esiste creo un file csv con i nomi delle colonne
                    if not os.path.exists('configurationTable.csv'):
                        with open('configurationTable.csv', 'a+') as f:
                            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                            writer.writerow(['class'] + [f'{name}_x' for name in landmarks_name] + [f'{name}_y' for name in landmarks_name] + [f'{name}_z' for name in landmarks_name] + [f'{name}_visibility' for name in landmarks_name] + ['crouch', 'hands_distance', 'hands_visibility'
                                                                                                                                                                                                                , 'body_direction', 'head_direction' , 'hands_physical_distance', 'body_y', 'crouched_bound_triangle_custom_proportion', 'crouched_yTorso_custom_proportion'
                                                                                                                                                                                                                    ] )
                    ai_data = list(np.array( [dataTable[f'{name}_x'][-1] for name in landmarks_name] + [dataTable[f'{name}_y'][-1] for name in landmarks_name] + [dataTable[f'{name}_z'][-1] for name in landmarks_name] +
                                            [dataTable[f'{name}_visibility'][-1] for name in landmarks_name] + [featuresTable[f'crouch'][-1], featuresTable[f'hands_distance'][-1], featuresTable[f'hands_visibility'][-1]
                                            , featuresTable[f'head_direction'][-1] , featuresTable[f'body_direction'][-1], featuresTable[f'hands_physical_distance'][-1], featuresTable[f'body_y'][-1], featuresTable[f'crouched_bound_triangle_custom_proportion'][-1], featuresTable[f'crouched_yTorso_custom_proportion'][-1]                                                                  
                                                                                                                    ] ))
                    ai_data.insert(0, "CLASS") # to change with the class of the user !!!
                    with open('configurationTable.csv', 'a') as f:
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

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
    saveCalibrationData(dataTable, featuresTable, 'configurationTable.csv', calibrationData, hands_calibration, crouch_calibration, body_direction_calibration, crouched_yTorso_custom_proportion, crouched_bound_triangle_custom_proportion)


if __name__ == "__main__":
    
    with open('bodyCalibrations.json') as f:
        calibrationData = json.load(f)

    # check if the input parameters are correct
    if len(sys.argv) < 5:
        print("Usage: python script.py crouch_calibration hands_calibration body_direction_calibration configuration_time")
        sys.exit(1)
    
    # Define the input parameters
    crouch_calibration = sys.argv[1].lower() == 'true'
    hands_calibration = sys.argv[2].lower() == 'true'
    body_direction_calibration = sys.argv[3].lower() == 'true'
    configuration_time = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    print("Input settings: \n", 
            "crouch_calibration: ", crouch_calibration, 
            "hands_calibration: ", hands_calibration, 
            "body_direction_calibration: ", body_direction_calibration, 
            "configuration_time: ", configuration_time)
    # start mediapipe
    mediaPipe(crouch_calibration, hands_calibration, body_direction_calibration, configuration_time, calibrationData)
  
