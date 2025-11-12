from picamera2 import Picamera2
import cv2
import numpy as np
import time

class PanoramicStitcher:
    def __init__(self):
        self.stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        self.is_calibrated = False
        self.H = None  # Homography matrix
        
    def calibrate(self, frame0, frame1):
        """Calibrate the stitcher with initial frames"""
        print("Calibrating stitcher...")
        status, stitched = self.stitcher.stitch([frame0, frame1])
        if status == cv2.Stitcher_OK:
            self.is_calibrated = True
            print("Calibration successful!")
            return True
        else:
            print(f"Calibration failed with status: {status}")
            return False
    
    def stitch(self, frame0, frame1):
        """Stitch frames using calibrated parameters"""
        if not self.is_calibrated:
            return None, cv2.Stitcher_ERR_NEED_MORE_IMGS
        
        status, stitched = self.stitcher.stitch([frame0, frame1])
        return stitched, status

# Initialize cameras
cam0 = Picamera2(0)
cam1 = Picamera2(1)

config0 = cam0.create_preview_configuration(main={"size": (1280, 720)})
config1 = cam1.create_preview_configuration(main={"size": (1280, 720)})

cam0.configure(config0)
cam1.configure(config1)

cam0.start()
cam1.start()

# Initialize stitcher
stitcher = PanoramicStitcher()

# FPS tracking
fps = 0
frame_count = 0
start_time = time.time()

# Warm up
time.sleep(2)

print("Hold cameras steady for calibration...")

try:
    calibration_done = False
    
    while True:
        frame0 = cam0.capture_array()
        frame1 = cam1.capture_array()
        
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
        
        # Calibrate on first successful capture
        if not calibration_done:
            if stitcher.calibrate(frame0, frame1):
                calibration_done = True
            else:
                # Show debug view during calibration
                debug = np.hstack((frame0, frame1))
                cv2.putText(debug, "Calibrating... Hold steady", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Calibration', debug)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
        
        # Stitch frames
        stitched, status = stitcher.stitch(frame0, frame1)
        
        if status == cv2.Stitcher_OK and stitched is not None:
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            cv2.putText(stitched, f'FPS: {fps:.2f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Panoramic View', stitched)
        else:
            print("Stitching error, recalibrating...")
            calibration_done = False
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    cam0.stop()
    cam1.stop()
    cv2.destroyAllWindows()
