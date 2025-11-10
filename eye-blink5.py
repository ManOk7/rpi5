import cv2
import mediapipe as mp
import numpy as np
import time

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

print("Initializing Raspberry Pi AI Camera...")

# Try Picamera2 first (preferred for AI Camera)
try:
    from picamera2 import Picamera2
    
    picam2 = Picamera2()
    
    # Configure for AI Camera with optimal settings
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        controls={"FrameRate": 30}
    )
    picam2.configure(config)
    picam2.start()
    
    # Wait for camera to stabilize
    time.sleep(2)
    
    use_picamera2 = True
    print("✓ AI Camera initialized with Picamera2")
    
except ImportError:
    print("Picamera2 not found, trying alternative methods...")
    use_picamera2 = False
    
    # Try libcamera GStreamer pipeline
    gst_pipeline = (
        "libcamerasrc ! "
        "video/x-raw,width=640,height=480,framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink"
    )
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("GStreamer failed, trying V4L2...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("\n❌ ERROR: Cannot open AI Camera!")
        print("\nPlease run these commands:")
        print("  sudo apt install -y imx500-all python3-picamera2")
        print("  sudo reboot")
        exit(1)
    
    print("✓ AI Camera initialized with OpenCV")

# FPS calculation
fps_time = time.time()
fps_counter = 0
fps = 0

print("\n" + "="*50)
print("EAR Blink Detection - Raspberry Pi AI Camera")
print("="*50)
print("\nControls:")
print("  ESC - Exit")
print("  C   - Show current EAR (for calibration)")
print("  R   - Reset blink count")
print("  +   - Increase threshold")
print("  -   - Decrease threshold")
print("\nStarting detection...\n")

try:
    while True:
        # Capture frame based on camera type
        if use_picamera2:
            frame = picam2.capture_array()
            # Convert RGB to BGR for OpenCV display
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Use original RGB frame for MediaPipe
            rgb_frame = frame
        else:
            ret, frame_bgr = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Get frame dimensions
        h, w, _ = frame_bgr.shape
        
        # Process face mesh
        results = face_mesh.process(rgb_frame)

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
                    cv2.putText(frame_bgr, "BLINK!", (w//2 - 80, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    if blink_counter >= CONSEC_FRAMES:
                        total_blinks += 1
                        print(f"Blink detected! Total: {total_blinks}")
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
        
        # AI Camera indicator
        cv2.putText(frame_bgr, "AI CAM", (w - 150, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display frame
        cv2.imshow("EAR Blink Detection - RPi AI Camera", frame_bgr)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == ord('c') or key == ord('C'):  # Calibrate threshold
            if results.multi_face_landmarks:
                print(f"\n--- Calibration Info ---")
                print(f"Current EAR: {ear:.3f}")
                print(f"Left Eye EAR: {left_ear:.3f}")
                print(f"Right Eye EAR: {right_ear:.3f}")
                print(f"Current threshold: {EAR_THRESHOLD:.2f}")
                print("Close your eyes and press 'C' to see closed eye EAR\n")
        elif key == ord('r') or key == ord('R'):  # Reset blink count
            total_blinks = 0
            print("Blink count reset to 0")
        elif key == ord('+') or key == ord('='):  # Increase threshold
            EAR_THRESHOLD += 0.01
            print(f"Threshold increased to: {EAR_THRESHOLD:.2f}")
        elif key == ord('-') or key == ord('_'):  # Decrease threshold
            EAR_THRESHOLD -= 0.01
            print(f"Threshold decreased to: {EAR_THRESHOLD:.2f}")

except KeyboardInterrupt:
    print("\n\nInterrupted by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Cleanup
    if use_picamera2:
        picam2.stop()
    else:
        cap.release()
    cv2.destroyAllWindows()
    print(f"\n{'='*50}")
    print(f"Total blinks detected: {total_blinks}")
    print(f"{'='*50}")
    print("Program terminated successfully")
