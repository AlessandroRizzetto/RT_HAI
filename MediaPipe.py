import mediapipe as mp
import cv2
import numpy as np
import os


# Setup MediaPipe Holistic instance
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Setup drawing utility
mp_drawing = mp.solutions.drawing_utils

# If exists previous data, delete it
if os.path.exists("landmarks.stream"):
    os.remove("landmarks.stream")
    print("landmarks.stream file removed")

# Make Detections
cap = cv2.VideoCapture(0)
start_time = None

with open("landmarks.stream", "a+") as f:
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
            # 0
            nose = [landmarks[mp_holistic.PoseLandmark.NOSE.value].x,
                    landmarks[mp_holistic.PoseLandmark.NOSE.value].y]
            # 13, 14
            left_elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
            # 15, 16
            left_wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
            # 11, 12
            left_shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
            # 23, 24
            left_hip = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]
            # 17, 18
            left_pinky = [landmarks[mp_holistic.PoseLandmark.LEFT_PINKY.value].x,
                          landmarks[mp_holistic.PoseLandmark.LEFT_PINKY.value].y]
            right_pinky = [landmarks[mp_holistic.PoseLandmark.RIGHT_PINKY.value].x,
                           landmarks[mp_holistic.PoseLandmark.RIGHT_PINKY.value].y]
            # 19, 20
            left_index = [landmarks[mp_holistic.PoseLandmark.LEFT_INDEX.value].x,
                          landmarks[mp_holistic.PoseLandmark.LEFT_INDEX.value].y]
            right_index = [landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX.value].x,
                           landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX.value].y]

            # print("nose: ", (1-nose[1]))

            # save data into .stream file

            f.write(str(1-nose[1]) + "\n")
            f.flush()
            os.fsync(f.fileno())  # flush the buffer

            # Extract Pose landmarks for AI sulution
            # pose = results.pose_landmarks.landmark
            # pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            # # Extract Face landmarks
            # face = results.face_landmarks.landmark
            # face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            # # Concate rows
            # row = pose_row+face_row

            # # Make Detections for AI solution
            # x = pd.DataFrame([row])
            # model_class = model.predict(x)[0] #class of the model
            # model_prob = model.predict_proba(x)[0] #probability of each class
            # #print(model_class, model_prob)
            # actual_stage = model_class

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
