import cv2
import numpy as np
from picamera2 import Picamera2
from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams
from scipy.spatial import distance
import time

class BlinkDetector:
    def __init__(self, hef_path="face_landmarks.hef"):
        """
        Initialize blink detector with Hailo acceleration
        Args:
            hef_path: Path to Hailo HEF model file for face landmarks
        """
        # EAR threshold and consecutive frames for blink detection
        self.EAR_THRESHOLD = 0.25
        self.CONSEC_FRAMES = 2
        
        # Counter and blink tracking
        self.counter = 0
        self.total_blinks = 0
        
        # Eye landmark indices (for 68-point facial landmarks)
        self.LEFT_EYE_POINTS = list(range(36, 42))
        self.RIGHT_EYE_POINTS = list(range(42, 48))
        
        # Initialize Hailo
        self.setup_hailo(hef_path)
        
        # Initialize camera
        self.setup_camera()
    
    def setup_hailo(self, hef_path):
        """Setup Hailo AI accelerator"""
        try:
            # Create VDevice and load HEF
            self.device = VDevice()
            self.hef = HEF(hef_path)
            
            # Configure network
            self.network_group = self.device.configure(self.hef)[0]
            self.network_group_params = self.network_group.create_params()
            
            print("Hailo AI accelerator initialized successfully")
        except Exception as e:
            print(f"Error initializing Hailo: {e}")
            print("Make sure you have the correct HEF model file")
            raise
    
    def setup_camera(self):
        """Setup Picamera2 for RPi5"""
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)  # Camera warm-up
        print("Camera initialized")
    
    def calculate_ear(self, eye_points):
        """
        Calculate Eye Aspect Ratio
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        # Vertical eye distances
        vertical1 = distance.euclidean(eye_points[1], eye_points[5])
        vertical2 = distance.euclidean(eye_points[2], eye_points[4])
        
        # Horizontal eye distance
        horizontal = distance.euclidean(eye_points[0], eye_points[3])
        
        # Calculate EAR
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
    
    def preprocess_frame(self, frame):
        """Preprocess frame for Hailo inference"""
        # Resize to model input size (typically 224x224 or 256x256)
        input_size = (224, 224)  # Adjust based on your model
        resized = cv2.resize(frame, input_size)
        
        # Normalize (adjust based on model requirements)
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        batch = np.expand_dims(normalized, axis=0)
        return batch
    
    def run_hailo_inference(self, frame):
        """Run inference on Hailo to get facial landmarks"""
        preprocessed = self.preprocess_frame(frame)
        
        with InferVStreams(self.network_group, self.network_group_params) as infer_pipeline:
            # Send frame to Hailo
            input_data = {self.hef.get_input_vstream_infos()[0].name: preprocessed}
            
            # Run inference
            output = infer_pipeline.infer(input_data)
            
            # Extract landmarks from output
            landmarks = self.extract_landmarks(output)
            return landmarks
    
    def extract_landmarks(self, output):
        """Extract facial landmarks from Hailo output"""
        # This depends on your specific model output format
        # Typically landmarks are returned as (x, y) coordinates
        output_name = list(output.keys())[0]
        landmarks = output[output_name][0]  # Remove batch dimension
        
        # Reshape to (num_points, 2) if needed
        if landmarks.shape[-1] != 2:
            landmarks = landmarks.reshape(-1, 2)
        
        return landmarks
    
    def detect_blink(self, frame):
        """Main blink detection logic"""
        # Get facial landmarks using Hailo
        landmarks = self.run_hailo_inference(frame)
        
        if landmarks is None or len(landmarks) < 48:
            return frame, None
        
        # Extract eye coordinates
        left_eye = landmarks[self.LEFT_EYE_POINTS]
        right_eye = landmarks[self.RIGHT_EYE_POINTS]
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        
        # Average EAR
        ear = (left_ear + right_ear) / 2.0
        
        # Check if EAR is below threshold (eye closed)
        if ear < self.EAR_THRESHOLD:
            self.counter += 1
        else:
            # If eyes were closed for sufficient frames, count as blink
            if self.counter >= self.CONSEC_FRAMES:
                self.total_blinks += 1
            self.counter = 0
        
        # Draw eye contours
        left_eye_hull = cv2.convexHull(left_eye.astype(np.int32))
        right_eye_hull = cv2.convexHull(right_eye.astype(np.int32))
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
        
        return frame, ear
    
    def run(self):
        """Main loop"""
        print("Starting blink detection... Press 'q' to quit")
        
        try:
            while True:
                # Capture frame
                frame = self.picam2.capture_array()
                
                # Detect blinks
                annotated_frame, ear = self.detect_blink(frame)
                
                # Display info
                if ear is not None:
                    cv2.putText(annotated_frame, f"EAR: {ear:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(annotated_frame, f"Blinks: {self.total_blinks}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if ear is not None and ear < self.EAR_THRESHOLD:
                    cv2.putText(annotated_frame, "BLINK!", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display frame
                cv2.imshow("Blink Detection", annotated_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.picam2.stop()
        cv2.destroyAllWindows()
        print(f"\nTotal blinks detected: {self.total_blinks}")


if __name__ == "__main__":
    # Initialize detector with your HEF model path
    # You need a facial landmarks detection model converted to HEF format
    detector = BlinkDetector(hef_path="path/to/your/face_landmarks.hef")
    detector.run()
