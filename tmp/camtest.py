from ultralytics import YOLO
import cv2
import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class CameraConfig:
	"""Configuration settings for camera operations."""
	frame_width: int = 640
	frame_height: int = 480
	fps: int = 30
	fourcc: str = 'MJPG'
	yolo_image_size: int = 320
	confidence_threshold: float = 0.5
	person_class_id: int = 0
	horizontal_fov: float = 60.0  # degrees

class AngleCalculator:
	"""Calculates the angle of an object from the camera's center."""

	def __init__(self, config: CameraConfig):
		self.config = config

	def calculate_angle(self, object_center_x: int) -> float:
		"""
		Calculate the horizontal angle of an object from the center of the camera's view.

		Args:
			object_center_x: The horizontal center of the detected object in pixels.

		Returns:
			The angle in degrees. Positive values are to the right of center,
			negative values are to the left.
		"""
		frame_center_x = self.config.frame_width / 2
		pixel_offset = object_center_x - frame_center_x
		angle_per_pixel = self.config.horizontal_fov / self.config.frame_width
		angle = pixel_offset * angle_per_pixel
		return angle

model = YOLO('yolo11n.pt')
model.to('cuda' if torch.cuda.is_available() else 'cpu')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit(1)

camera_config = CameraConfig()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.frame_height)
cap.set(cv2.CAP_PROP_FPS, camera_config.fps)
angle_calculator = AngleCalculator(camera_config)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))

# Dictionary to store the previous center points of tracked objects
prev_centers = {}
# Dictionary to store how many consecutive frames an object has been still
stillness_counters = {}
# Dictionary to store the anchor point when an object starts being still
anchor_points = {}
# Threshold for movement detection (in pixels)
MOVEMENT_THRESHOLD = 10 # Lowered for more sensitivity to slow movement
# Number of frames an object must be still to be considered not moving
STILLNESS_FRAME_LIMIT = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Use track to get object IDs. Filter for 'person' class (class 0)
    results = model.track(frame, imgsz=320, stream=True, verbose=False, conf=0.5, persist=True, classes=0)
    
    annotated_frame = frame.copy()
    current_centers = {}
    current_track_ids = set()

    # Iterate over the generator to get the results
    for r in results:
        boxes = r.boxes
        if boxes.id is not None:  # Check if tracking IDs are available
            track_ids = boxes.id.int().cpu().tolist()
            xyxys = boxes.xyxy.cpu().numpy()

            for track_id, xyxy in zip(track_ids, xyxys):
                current_track_ids.add(track_id)
                x1, y1, x2, y2 = map(int, xyxy)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                current_centers[track_id] = (cx, cy)

                angle = angle_calculator.calculate_angle(cx)

                if track_id in prev_centers:
                    prev_cx, prev_cy = prev_centers[track_id]
                    distance = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)

                    # Check drift from anchor point
                    drift_distance = 0
                    if track_id in anchor_points:
                        anchor_cx, anchor_cy = anchor_points[track_id]
                        drift_distance = np.sqrt((cx - anchor_cx)**2 + (cy - anchor_cy)**2)

                    if distance < MOVEMENT_THRESHOLD and drift_distance < MOVEMENT_THRESHOLD:
                        # If not anchored, set anchor point
                        if track_id not in anchor_points:
                            anchor_points[track_id] = (cx, cy)
                        # Increment stillness counter if movement is below threshold
                        stillness_counters[track_id] = stillness_counters.get(track_id, 0) + 1
                    else:
                        # Reset counter and anchor if movement is detected
                        stillness_counters[track_id] = 0
                        if track_id in anchor_points:
                            del anchor_points[track_id]
                else:
                    # New track, assume moving until proven still
                    stillness_counters[track_id] = 0

                # Annotate if the object is not considered still yet
                if stillness_counters.get(track_id, 0) < STILLNESS_FRAME_LIMIT:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"Moving Person: {track_id} Angle: {angle:.2f}"
                    cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Clean up old tracks from stillness_counters and anchor_points
    stale_tracks = set(stillness_counters.keys()) - current_track_ids
    for track_id in stale_tracks:
        del stillness_counters[track_id]
        if track_id in anchor_points:
            del anchor_points[track_id]

    # Update previous centers for the next frame
    prev_centers = current_centers

    cv2.imshow('YOLO Camera', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
