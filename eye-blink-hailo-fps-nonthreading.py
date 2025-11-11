from picamera2 import Picamera2
import cv2
import numpy as np
import time

# Initialize both CSI cameras
cam0 = Picamera2(0)
cam1 = Picamera2(1)

# Configure cameras
config0 = cam0.create_preview_configuration(main={"size": (640, 480)})
config1 = cam1.create_preview_configuration(main={"size": (640, 480)})

cam0.configure(config0)
cam1.configure(config1)

# Start cameras
cam0.start()
cam1.start()

# FPS calculation variables
fps = 0
frame_count = 0
start_time = time.time()

try:
    while True:
        # Capture frames
        frame0 = cam0.capture_array()
        frame1 = cam1.capture_array()
        
        # Convert from RGB to BGR for OpenCV
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
        
        # Simple horizontal stitch
        stitched = np.hstack((frame0, frame1))
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:  # Update FPS every second
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Display FPS on the frame
        cv2.putText(stitched, f'FPS: {fps:.2f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Stitched CSI Cameras', stitched)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cam0.stop()
    cam1.stop()
    cv2.destroyAllWindows()
