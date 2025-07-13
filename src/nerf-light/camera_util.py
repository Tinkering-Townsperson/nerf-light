from ultralytics import YOLO
import cv2
import torch

class OptimizedCamera:
    def __init__(self, model_path: str, source: int = 0, debug: bool = False):
        """
        Initializes the Camera object.

        Args:
            model_path (str): Path to the YOLO model file (e.g., 'yolov8n.pt').
            source (int): Camera source index.
            debug (bool): If True, displays the video feed with detections.
        """
        self.source = source
        self.debug = debug
        
        # Check for GPU availability and select device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load the YOLO model
        self.model = YOLO(model_path)
        # Move model to the selected device
        self.model.to(self.device)

    def mainloop(self):
        """
        The main loop for capturing and processing video frames.
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.source}")
            return
            
        # Optional: Set camera properties. Note: These might not be supported by all cameras.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame, ending loop.")
                    break

                # --- OPTIMIZED INFERENCE ---
                # Use stream=True for efficient video processing.
                # Filter for 'person' class (class 0) directly in the predict call.
                results = self.model.predict(
                    frame, 
                    stream=True, 
                    conf=0.5, 
                    classes=0, # Only detect 'person' class
                    verbose=False # Suppress verbose output
                )

                # The results generator will yield one Result object per frame
                for r in results:
                    # Use the built-in .plot() method to draw detections
                    annotated_frame = r.plot()

                    # Process boxes for custom logic
                    for box in r.boxes:
                        self.parse_box(box)
                
                if self.debug:
                    # Show the frame with annotations
                    cv2.imshow("Optimized Camera", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except Exception as e:
            print(f"An error occurred in the mainloop: {e}")
        finally:
            print("Releasing resources.")
            cap.release()
            cv2.destroyAllWindows()

    def parse_box(self, box):
        """
        Parses a single bounding box and prints its information.
        """
        # box.xyxy[0] is a tensor of shape [4], representing [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf)
        class_id = int(box.cls)
        class_name = self.model.names[class_id]
        
        print(f"{class_name} detected with confidence: {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")

# --- HOW TO RUN ---
if __name__ == '__main__':
    # Use the fastest model available
    MODEL_PATH = 'yolov8n.pt'
    
    # Create and run the camera
    # Set debug=True to see the video feed
    camera_detector = OptimizedCamera(model_path=MODEL_PATH, debug=True)
    camera_detector.mainloop()

# Alias for compatibility with code expecting 'Camera'
Camera = OptimizedCamera