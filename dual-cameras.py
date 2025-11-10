from picamera2 import Picamera2
import cv2
import time

# Initialize both cameras
cam0 = Picamera2(camera_num=0)
cam1 = Picamera2(camera_num=1)

config0 = cam0.create_preview_configuration(main={"size": (640, 480)})
config1 = cam1.create_preview_configuration(main={"size": (640, 480)})

cam0.configure(config0)
cam1.configure(config1)

cam0.start()
cam1.start()

time.sleep(1)

while True:
    frame0 = cam0.capture_array()
    frame1 = cam1.capture_array()

    cv2.imshow("Camera 0", frame0)
    cv2.imshow("Camera 1", frame1)

    if cv2.waitKey(1) == ord('q'):
        break

cam0.stop()
cam1.stop()
cv2.destroyAllWindows()
