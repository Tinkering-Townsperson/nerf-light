from ultralytics import YOLO
import cv2
import torch
import numpy as np


class Camera:
	def __init__(self, model_path: str = 'yolo11n.pt', source: int = 0, debug: bool = False):
		"""
		Initializes the Camera object.

		Args:
			model_path (str): Path to the YOLO model file (e.g., 'yolo11n.pt').
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

		# Movement detection parameters
		self.prev_centers = {}
		self.stillness_counters = {}
		self.anchor_points = {}
		self.MOVEMENT_THRESHOLD = 10  # pixels
		self.STILLNESS_FRAME_LIMIT = 5  # frames

	def mainloop(self):
		"""
		The main loop for capturing and processing video frames.
		"""
		cap = cv2.VideoCapture(self.source)
		if not cap.isOpened():
			print(f"Error: Could not open video source {self.source}")
			return

		# Set camera properties for optimal performance
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

				# Use track method to get object IDs and filter for 'person' class
				results = self.model.track(
					frame,
					imgsz=320,
					stream=True,
					verbose=False,
					conf=0.5,
					persist=True,
					classes=0  # Person class
				)

				annotated_frame = frame.copy()
				current_centers = {}
				current_track_ids = set()

				# Process tracking results
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

							# Process movement detection for this tracked object
							is_moving = self._process_movement_detection(track_id, cx, cy)

							# Annotate moving objects
							if is_moving:
								cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
								cv2.putText(
									annotated_frame,
									f"Moving Person: {track_id}",
									(x1, y1 - 10),
									cv2.FONT_HERSHEY_SIMPLEX,
									0.5,
									(0, 255, 0),
									2
								)

							# Custom processing for each detection
							self.parse_tracked_object(track_id, x1, y1, x2, y2, is_moving)

				# Clean up stale tracks
				self._cleanup_stale_tracks(current_track_ids)

				# Update previous centers for next frame
				self.prev_centers = current_centers

				if self.debug:
					cv2.imshow('YOLO Camera with Movement Detection', annotated_frame)
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break

		except Exception as e:
			print(f"An error occurred in the mainloop: {e}")
		finally:
			print("Releasing resources.")
			cap.release()
			cv2.destroyAllWindows()

	def _process_movement_detection(self, track_id: int, cx: int, cy: int) -> bool:
		"""
		Process movement detection for a tracked object.
		
		Args:
			track_id (int): Unique identifier for the tracked object
			cx (int): Center x coordinate
			cy (int): Center y coordinate
			
		Returns:
			bool: True if the object is considered moving, False otherwise
		"""
		if track_id in self.prev_centers:
			prev_cx, prev_cy = self.prev_centers[track_id]
			distance = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)

			# Check drift from anchor point
			drift_distance = 0
			if track_id in self.anchor_points:
				anchor_cx, anchor_cy = self.anchor_points[track_id]
				drift_distance = np.sqrt((cx - anchor_cx)**2 + (cy - anchor_cy)**2)

			if distance < self.MOVEMENT_THRESHOLD and drift_distance < self.MOVEMENT_THRESHOLD:
				# If not anchored, set anchor point
				if track_id not in self.anchor_points:
					self.anchor_points[track_id] = (cx, cy)
				# Increment stillness counter
				self.stillness_counters[track_id] = self.stillness_counters.get(track_id, 0) + 1
			else:
				# Reset counter and anchor if movement is detected
				self.stillness_counters[track_id] = 0
				if track_id in self.anchor_points:
					del self.anchor_points[track_id]
		else:
			# New track, assume moving until proven still
			self.stillness_counters[track_id] = 0

		# Return True if object is considered moving
		return self.stillness_counters.get(track_id, 0) < self.STILLNESS_FRAME_LIMIT

	def _cleanup_stale_tracks(self, current_track_ids: set):
		"""
		Clean up tracking data for objects that are no longer being tracked.
		
		Args:
			current_track_ids (set): Set of currently active track IDs
		"""
		stale_tracks = set(self.stillness_counters.keys()) - current_track_ids
		for track_id in stale_tracks:
			del self.stillness_counters[track_id]
			if track_id in self.anchor_points:
				del self.anchor_points[track_id]
			if track_id in self.prev_centers:
				del self.prev_centers[track_id]

	def parse_tracked_object(self, track_id: int, x1: int, y1: int, x2: int, y2: int, is_moving: bool):
		"""
		Parses a tracked object and handles custom logic.
		
		Args:
			track_id (int): Unique identifier for the tracked object
			x1, y1, x2, y2 (int): Bounding box coordinates
			is_moving (bool): Whether the object is currently moving
		"""
		if is_moving:
			print(f"Moving person {track_id} detected at [{x1}, {y1}, {x2}, {y2}]")
		else:
			print(f"Still person {track_id} detected at [{x1}, {y1}, {x2}, {y2}]")

	def parse_box(self, box):
		"""
		Legacy method for compatibility - parses a single bounding box.
		"""
		# box.xyxy[0] is a tensor of shape [4], representing [x1, y1, x2, y2]
		x1, y1, x2, y2 = map(int, box.xyxy[0])
		confidence = float(box.conf)
		class_id = int(box.cls)
		class_name = self.model.names[class_id]

		print(f"{class_name} detected with confidence: {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")
