from picamera2 import Picamera2
import mediapipe as mp
import numpy as np
import cv2
import time

# Initialize picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(1)

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# EAR calculation function
def calculate_ear(landmarks, eye_indices, w, h):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    ear = (A + B) / (2.0 * C)
    return ear, points

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

while True:
    frame = picam2.capture_array()
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            left_ear, left_points = calculate_ear(face.landmark, LEFT_EYE, w, h)
            right_ear, right_points = calculate_ear(face.landmark, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            for p in left_points + right_points:
                cv2.circle(frame, p, 2, (0, 255, 0), -1)

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            if ear < 0.20:
                cv2.putText(frame, "BLINK", (200, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    cv2.imshow("EAR Detection - Raspberry Pi 5", frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
picam2.stop()
