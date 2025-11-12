from picamera2 import Picamera2
import cv2
import numpy as np
import time

# Initialize both CSI cameras
cam0 = Picamera2(0)
cam1 = Picamera2(1)

# Configure cameras
config0 = cam0.create_preview_configuration(main={"size": (1280, 720)})
config1 = cam1.create_preview_configuration(main={"size": (1280, 720)})

cam0.configure(config0)
cam1.configure(config1)

# Start cameras
cam0.start()
cam1.start()

# Create stitcher object
# PANORAMA mode works best for side-by-side cameras
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

# FPS calculation
fps = 0
frame_count = 0
start_time = time.time()

# Warm up cameras
time.sleep(2)

print("Starting panoramic stitching...")
print("Press 'q' to quit")

try:
    while True:
        # Capture frames
        frame0 = cam0.capture_array()
        frame1 = cam1.capture_array()
        
        # Convert from RGB to BGR for OpenCV
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
        
        # Stitch the frames together
        status, stitched = stitcher.stitch([frame0, frame1])
        
        if status == cv2.Stitcher_OK:
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Display FPS
            cv2.putText(stitched, f'FPS: {fps:.2f}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Panoramic Stitch', stitched)
        else:
            print(f"Stitching failed with status: {status}")
            # Show original frames side by side for debugging
            debug = np.hstack((frame0, frame1))
            cv2.imshow('Debug - Original Frames', debug)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    cam0.stop()
    cam1.stop()
    cv2.destroyAllWindows()
