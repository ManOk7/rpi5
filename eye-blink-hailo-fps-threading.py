from picamera2 import Picamera2
import cv2
import numpy as np
from threading import Thread
import time

class CSICamera:
    def __init__(self, camera_num):
        self.cam = Picamera2(camera_num)
        config = self.cam.create_preview_configuration(main={"size": (640, 480)})
        self.cam.configure(config)
        self.cam.start()
        self.frame = None
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            self.frame = self.cam.capture_array()
            
    def read(self):
        return self.frame
        
    def stop(self):
        self.stopped = True
        self.cam.stop()

# Initialize and start both cameras
cam0 = CSICamera(0).start()
cam1 = CSICamera(1).start()

# FPS calculation variables
fps = 0
frame_count = 0
start_time = time.time()

# Wait a moment for cameras to warm up
time.sleep(2)

try:
    while True:
        frame0 = cam0.read()
        frame1 = cam1.read()
        
        if frame0 is None or frame1 is None:
            continue
            
        # Convert RGB to BGR
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
        
        # Stitch
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
