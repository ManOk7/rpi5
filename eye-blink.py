import cv2
import mediapipe as mp
import numpy as np

# ======= SELECT CAMERA HERE =======
CAMERA_INDEX = 0   # change to 1 for second camera
# ==================================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_ear(landmarks, eye_indices, w, h):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    ear = (A + B) / (2.0 * C)
    return ear, points

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not available. Try changing CAMERA_INDEX at top.")
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear, left_pts = calculate_ear(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear, right_pts = calculate_ear(face_landmarks.landmark, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            for p in left_pts + right_pts:
                cv2.circle(frame, p, 2, (0,255,0), -1)

            cv2.putText(frame, f"EAR: {ear:.2f}", (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if ear < 0.20:
                cv2.putText(frame, "DROWSY!", (200,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3)

    cv2.imshow("Raspberry Pi EAR Drowsiness", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
