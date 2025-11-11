import cv2
import numpy as np
from picamera2 import Picamera2
from hailo_platform import (VDevice, HEF, ConfigureParams, 
                            InferVStreams, InputVStreamParams, 
                            OutputVStreamParams, FormatType)
from scipy.spatial import distance
import time

class BlinkDetector:
    def __init__(self, hef_path="face_landmarks.hef", camera_num=0):
        """
        Initialize blink detector with Hailo acceleration (HailoRT v4.20)
        Args:
            hef_path: Path to Hailo HEF model file for face landmarks
            camera_num: Camera index (0 or 1 for dual camera setup)
        """
        self.camera_num = camera_num
        self.hef_path = hef_path
        
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
        self.setup_hailo()
        
        # Initialize camera
        self.setup_camera()
    
    def setup_hailo(self):
        """Setup Hailo AI accelerator with HailoRT v4.20 API"""
        try:
            print("Initializing Hailo AI accelerator...")
            
            # Create VDevice (target device)
            self.target = VDevice()
            
            # Load HEF file
            self.hef = HEF(self.hef_path)
            
            # Configure the device with the HEF
            configure_params = ConfigureParams.create_from_hef(self.hef, interface=FormatType.FLOAT32)
            self.network_group = self.target.configure(self.hef, configure_params)[0]
            
            # Get input and output virtual stream parameters
            self.input_vstreams_params = InputVStreamParams.make_from_network_group(
                self.network_group, quantized=False, format_type=FormatType.FLOAT32
            )
            self.output_vstreams_params = OutputVStreamParams.make_from_network_group(
                self.network_group, quantized=False, format_type=FormatType.FLOAT32
            )
            
            # Get input/output info
            self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
            self.output_vstream_info = self.hef.get_output_vstream_infos()[0]
            
            # Get input shape
            self.input_shape = self.input_vstream_info.shape
            self.input_height = self.input_shape[0]
            self.input_width = self.input_shape[1]
            
            print(f"✓ Hailo initialized successfully")
            print(f"  Model input shape: {self.input_shape}")
            print(f"  Model input name: {self.input_vstream_info.name}")
            print(f"  Model output name: {self.output_vstream_info.name}")
            
        except Exception as e:
            print(f"✗ Error initializing Hailo: {e}")
            print("\nTroubleshooting tips:")
            print("  1. Check HEF file path is correct")
            print("  2. Verify Hailo driver: hailortcli fw-control identify")
            print("  3. Ensure HailoRT v4.20+ is installed: pip3 show hailo-platform")
            raise
    
    def setup_camera(self):
        """Setup Picamera2 for RPi5 with camera selection"""
        # List all available cameras
        cameras = Picamera2.global_camera_info()
        print(f"\nAvailable cameras: {len(cameras)}")
        for idx, cam in enumerate(cameras):
            print(f"  Camera {idx}: {cam}")
        
        # Initialize the selected camera
        self.picam2 = Picamera2(self.camera_num)
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)  # Camera warm-up
        print(f"✓ Camera {self.camera_num} initialized successfully\n")
    
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
        
        # Avoid division by zero
        if horizontal == 0:
            return 0
        
        # Calculate EAR
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
    
    def preprocess_frame(self, frame):
        """Preprocess frame for Hailo inference"""
        # Resize to model input size
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Convert BGR to RGB if needed
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] (adjust based on your model's requirements)
        normalized = resized.astype(np.float32) / 255.0
        
        # Some models expect values in [-1, 1]
        # normalized = (resized.astype(np.float32) / 127.5) - 1.0
        
        return normalized
    
    def run_hailo_inference(self, frame):
        """Run inference on Hailo to get facial landmarks (HailoRT v4.20)"""
        try:
            # Preprocess frame
            input_data = self.preprocess_frame(frame)
            
            # Create input dictionary
            input_dict = {self.input_vstream_info.name: input_data}
            
            # Run inference with InferVStreams context manager
            with InferVStreams(self.network_group, 
                             self.input_vstreams_params, 
                             self.output_vstreams_params) as infer_pipeline:
                
                # Infer
                with self.network_group.activate():
                    infer_results = infer_pipeline.infer(input_dict)
            
            # Extract landmarks from results
            landmarks = self.extract_landmarks(infer_results, frame.shape)
            return landmarks
            
        except Exception as e:
            print(f"Inference error: {e}")
            return None
    
    def extract_landmarks(self, infer_results, original_shape):
        """Extract facial landmarks from Hailo output"""
        try:
            # Get output tensor
            output_name = self.output_vstream_info.name
            output_data = infer_results[output_name]
            
            # Output format depends on your model
            # Common formats:
            # 1. [num_landmarks * 2] - flattened (x1, y1, x2, y2, ...)
            # 2. [num_landmarks, 2] - structured
            # 3. [1, num_landmarks, 2] - with batch dimension
            
            # Remove batch dimension if present
            if len(output_data.shape) == 3:
                output_data = output_data[0]
            
            # Reshape to (num_points, 2) if flattened
            if len(output_data.shape) == 1:
                output_data = output_data.reshape(-1, 2)
            
            # Scale landmarks from normalized coordinates to pixel coordinates
            # Assuming model outputs normalized coordinates [0, 1]
            landmarks = output_data.copy()
            landmarks[:, 0] *= original_shape[1]  # Scale x coordinates
            landmarks[:, 1] *= original_shape[0]  # Scale y coordinates
            
            return landmarks.astype(np.int32)
            
        except Exception as e:
            print(f"Landmark extraction error: {e}")
            return None
    
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
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
        
        # Draw all facial landmarks (optional - for debugging)
        # for point in landmarks:
        #     cv2.circle(frame, tuple(point), 1, (0, 0, 255), -1)
        
        return frame, ear
    
    def run(self):
        """Main loop"""
        print("Starting blink detection...")
        print("Press 'q' to quit\n")
        
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        try:
            while True:
                # Capture frame
                frame = self.picam2.capture_array()
                
                # Detect blinks
                annotated_frame, ear = self.detect_blink(frame)
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter >= 30:
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    fps_counter = 0
                
                # Display info
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                if ear is not None:
                    cv2.putText(annotated_frame, f"EAR: {ear:.3f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.putText(annotated_frame, f"Blinks: {self.total_blinks}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if ear is not None and ear < self.EAR_THRESHOLD:
                    cv2.putText(annotated_frame, "BLINKING!", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display frame
                cv2.imshow("Blink Detection - Hailo Accelerated", annotated_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        self.picam2.stop()
        cv2.destroyAllWindows()
        print(f"Total blinks detected: {self.total_blinks}")
        print("Done!")


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Blink Detection with Hailo on RPi5 (HailoRT v4.20)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  List cameras:          python3 %(prog)s --list-cameras
  Use camera 0:          python3 %(prog)s --camera 0 --hef model.hef
  Use camera 1:          python3 %(prog)s --camera 1 --hef model.hef
  Adjust EAR threshold:  python3 %(prog)s --threshold 0.23
        """
    )
    parser.add_argument('--camera', type=int, default=0, 
                        help='Camera index (0 or 1 for dual camera setup)')
    parser.add_argument('--hef', type=str, default='face_landmarks.hef',
                        help='Path to HEF model file')
    parser.add_argument('--threshold', type=float, default=0.25,
                        help='EAR threshold for blink detection (default: 0.25)')
    parser.add_argument('--list-cameras', action='store_true',
                        help='List available cameras and exit')
    
    args = parser.parse_args()
    
    # List cameras if requested
    if args.list_cameras:
        cameras = Picamera2.global_camera_info()
        print(f"\nFound {len(cameras)} camera(s):")
        for idx, cam in enumerate(cameras):
            print(f"  Camera {idx}: {cam}")
        exit(0)
    
    # Initialize detector with specified camera
    print(f"=== Hailo Blink Detection (HailoRT v4.20) ===\n")
    detector = BlinkDetector(hef_path=args.hef, camera_num=args.camera)
    detector.EAR_THRESHOLD = args.threshold
    detector.run()
