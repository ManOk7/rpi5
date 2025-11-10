import cv2
import mediapipe as mp
import numpy as np
import time
from picamera2 import Picamera2

# Initialize mediapipe face mesh with optimized settings for RPi
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1,
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Function to calculate EAR (Eye Aspect Ratio)
def calculate_ear(landmarks, eye_indices, image_width, image_height):
    """Calculate Eye Aspect Ratio for blink detection"""
    # Convert normalized coordinates to pixel coordinates
    points = np.array([
        [landmarks[i].x * image_width, landmarks[i].y * image_height] 
        for i in eye_indices
    ])

    # EAR formula: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    
    ear = (A + B) / (2.0 * C + 1e-6)  # Add epsilon to avoid division by zero
    return ear, points.astype(int)

# Indices for left and right eye (MediaPipe FaceMesh 468-point model)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Blink detection parameters
EAR_THRESHOLD = 0.21  # Adjust based on your calibration
CONSEC_FRAMES = 2     # Number of consecutive frames for blink detection
blink_counter = 0
total_blinks = 0

# Initialize Picamera2 for CSI camera
print("Initializing CSI camera...")
picam2 = Picamera2()

# Configure camera for optimal performance
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={"FrameRate": 30}
)
picam2.configure(config)
picam2.start()

# Wait for camera to warm up
time.sleep(2)

# FPS calculation
fps_time = time.time()
fps_counter = 0
fps = 0

print("Starting EAR Detection with CSI Camera...")
print("Press 'ESC' to exit, 'C' to calibrate threshold, 'R' to reset blink count")

try:
    while True:
        # Capture frame from CSI camera
        frame = picam2.capture_array()
        
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Process face mesh (frame is already in RGB format)
        results = face_mesh.process(frame)
        
        # Convert to BGR for OpenCV display
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Calculate EAR for both eyes
                left_ear, left_points = calculate_ear(face_landmarks.landmark, LEFT_EYE, w, h)
                right_ear, right_points = calculate_ear(face_landmarks.landmark, RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0

                # Draw eye landmarks
                for p in left_points:
                    cv2.circle(frame_bgr, tuple(p), 2, (0, 255, 0), -1)
                for p in right_points:
                    cv2.circle(frame_bgr, tuple(p), 2, (0, 255, 0), -1)

                # Blink detection logic
                if ear < EAR_THRESHOLD:
                    blink_counter += 1
                    cv2.putText(frame_bgr, "BLINK DETECTED", (w//2 - 150, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    if blink_counter >= CONSEC_FRAMES:
                        total_blinks += 1
                    blink_counter = 0

                # Display EAR value
                cv2.putText(frame_bgr, f'EAR: {ear:.3f}', (30, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                
                # Display blink count
                cv2.putText(frame_bgr, f'Blinks: {total_blinks}', (30, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                
                # Display threshold
                cv2.putText(frame_bgr, f'Threshold: {EAR_THRESHOLD:.2f}', (30, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Calculate and display FPS
        fps_counter += 1
        if time.time() - fps_time > 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_time = time.time()
        
        cv2.putText(frame_bgr, f'FPS: {fps}', (w - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # CSI Camera indicator
        cv2.putText(frame_bgr, "CSI CAM", (w - 150, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display frame
        cv2.imshow("EAR Blink Detection - RPi5 CSI", frame_bgr)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == ord('c') or key == ord('C'):  # Calibrate threshold
            if results.multi_face_landmarks:
                print(f"Current EAR: {ear:.3f}")
                print("Current threshold: {:.2f}".format(EAR_THRESHOLD))
                print("Close your eyes and press 'C' again to see closed eye EAR")
        elif key == ord('r') or key == ord('R'):  # Reset blink count
            total_blinks = 0
            print("Blink count reset")
        elif key == ord('+') or key == ord('='):  # Increase threshold
            EAR_THRESHOLD += 0.01
            print(f"Threshold increased to: {EAR_THRESHOLD:.2f}")
        elif key == ord('-') or key == ord('_'):  # Decrease threshold
            EAR_THRESHOLD -= 0.01
            print(f"Threshold decreased to: {EAR_THRESHOLD:.2f}")

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    # Cleanup
    picam2.stop()
    cv2.destroyAllWindows()
    print(f"\nTotal blinks detected: {total_blinks}")
    print("Program terminated successfully")
