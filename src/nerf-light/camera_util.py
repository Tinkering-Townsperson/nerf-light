from ultralytics import YOLO
import cv2
import torch
import numpy as np
from typing import Dict, Set, Tuple, Optional, Any
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


@dataclass
class MovementConfig:
	"""Configuration settings for movement detection."""
	movement_threshold: int = 10  # pixels
	stillness_frame_limit: int = 5  # frames


class MovementDetector:
	"""Handles movement detection for tracked objects."""

	def __init__(self, config: MovementConfig):
		self.config = config
		self._previous_centers: Dict[int, Tuple[int, int]] = {}
		self._stillness_counters: Dict[int, int] = {}
		self._anchor_points: Dict[int, Tuple[int, int]] = {}

	def is_object_moving(self, track_id: int, center_x: int, center_y: int) -> bool:
		"""
		Determine if a tracked object is currently moving.

		Args:
			track_id: Unique identifier for the tracked object
			center_x: Current center x coordinate
			center_y: Current center y coordinate

		Returns:
			True if the object is considered moving, False otherwise
		"""
		if self._is_new_track(track_id):
			self._initialize_new_track(track_id)
			return True

		movement_distance = self._calculate_movement_distance(track_id, center_x, center_y)
		drift_distance = self._calculate_drift_distance(track_id, center_x, center_y)

		if self._is_within_movement_threshold(movement_distance, drift_distance):
			self._handle_stationary_object(track_id, center_x, center_y)
		else:
			self._handle_moving_object(track_id)

		return self._stillness_counters.get(track_id, 0) < self.config.stillness_frame_limit

	def update_centers(self, current_centers: Dict[int, Tuple[int, int]]) -> None:
		"""Update the previous centers for the next frame."""
		self._previous_centers = current_centers.copy()

	def cleanup_stale_tracks(self, active_track_ids: Set[int]) -> None:
		"""Remove tracking data for objects no longer being tracked."""
		stale_tracks = set(self._stillness_counters.keys()) - active_track_ids

		for track_id in stale_tracks:
			self._remove_track_data(track_id)

	def _is_new_track(self, track_id: int) -> bool:
		"""Check if this is a new track."""
		return track_id not in self._previous_centers

	def _initialize_new_track(self, track_id: int) -> None:
		"""Initialize tracking data for a new track."""
		self._stillness_counters[track_id] = 0

	def _calculate_movement_distance(self, track_id: int, center_x: int, center_y: int) -> float:
		"""Calculate distance moved since previous frame."""
		prev_x, prev_y = self._previous_centers[track_id]
		return np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)

	def _calculate_drift_distance(self, track_id: int, center_x: int, center_y: int) -> float:
		"""Calculate drift distance from anchor point."""
		if track_id not in self._anchor_points:
			return 0

		anchor_x, anchor_y = self._anchor_points[track_id]
		return np.sqrt((center_x - anchor_x)**2 + (center_y - anchor_y)**2)

	def _is_within_movement_threshold(self, movement_distance: float, drift_distance: float) -> bool:
		"""Check if movement is within threshold."""
		return (
			movement_distance < self.config.movement_threshold and
			drift_distance < self.config.movement_threshold
		)

	def _handle_stationary_object(self, track_id: int, center_x: int, center_y: int) -> None:
		"""Handle logic for stationary objects."""
		if track_id not in self._anchor_points:
			self._anchor_points[track_id] = (center_x, center_y)

		self._stillness_counters[track_id] = self._stillness_counters.get(track_id, 0) + 1

	def _handle_moving_object(self, track_id: int) -> None:
		"""Handle logic for moving objects."""
		self._stillness_counters[track_id] = 0
		if track_id in self._anchor_points:
			del self._anchor_points[track_id]

	def _remove_track_data(self, track_id: int) -> None:
		"""Remove all tracking data for a specific track."""
		self._stillness_counters.pop(track_id, None)
		self._anchor_points.pop(track_id, None)
		self._previous_centers.pop(track_id, None)


class FrameAnnotator:
	"""Handles frame annotation for detected objects."""

	# Constants for annotation styling
	MOVING_OBJECT_COLOR = (0, 255, 0)  # Green for moving objects
	FONT = cv2.FONT_HERSHEY_SIMPLEX
	FONT_SCALE = 0.5
	THICKNESS = 2
	TEXT_OFFSET_Y = -10

	@staticmethod
	def annotate_moving_object(frame: np.ndarray, track_id: int, x1: int, y1: int, x2: int, y2: int) -> None:
		"""Annotate a moving object on the frame."""
		cv2.rectangle(frame, (x1, y1), (x2, y2), FrameAnnotator.MOVING_OBJECT_COLOR, FrameAnnotator.THICKNESS)
		cv2.putText(
			frame,
			f"Moving Person: {track_id}",
			(x1, y1 + FrameAnnotator.TEXT_OFFSET_Y),
			FrameAnnotator.FONT,
			FrameAnnotator.FONT_SCALE,
			FrameAnnotator.MOVING_OBJECT_COLOR,
			FrameAnnotator.THICKNESS
		)


class Camera:
	"""Main camera class for object detection and tracking."""

	def __init__(
		self,
		handler: callable,
		model_path: str = 'yolo11n.pt',
		source: int = 0,
		debug: bool = False,
		weapon: Optional[Any] = None,
	):
		"""
		Initialize the Camera with YOLO model and tracking components.

		Args:
			model_path: Path to the YOLO model file
			source: Camera source index
			debug: If True, displays the video feed with detections
			weapon: An optional weapon object to control.
		"""
		self.source = source
		self.debug = debug
		self.camera_config = CameraConfig()
		self.movement_config = MovementConfig()
		self.weapon = weapon
		self.handler = handler

		self.device = self._initialize_device()
		self.model = self._load_model(model_path)

		# Initialize components following SRP
		self.movement_detector = MovementDetector(self.movement_config)
		self.annotator = FrameAnnotator()

	def _initialize_device(self) -> str:
		"""Initialize and return the appropriate device (GPU/CPU)."""
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		print(f"Using device: {device}")
		return device

	def _load_model(self, model_path: str) -> YOLO:
		"""Load and configure the YOLO model."""
		model = YOLO(model_path)
		model.to(self.device)
		return model

	def mainloop(self) -> None:
		"""Main loop for capturing and processing video frames."""
		cap = self._initialize_camera()
		if cap is None:
			return

		try:
			self._process_video_stream(cap)
		except Exception as e:
			print(f"An error occurred in the mainloop: {e}")
		finally:
			self._cleanup_resources(cap)

	def _initialize_camera(self) -> Optional[cv2.VideoCapture]:
		"""Initialize camera with optimal settings."""
		cap = cv2.VideoCapture(self.source)
		if not cap.isOpened():
			print(f"Error: Could not open video source {self.source}")
			return None

		self._configure_camera_properties(cap)
		return cap

	def _configure_camera_properties(self, cap: cv2.VideoCapture) -> None:
		"""Configure camera properties for optimal performance."""
		cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*self.camera_config.fourcc))
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.frame_width)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.frame_height)
		cap.set(cv2.CAP_PROP_FPS, self.camera_config.fps)

	def _process_video_stream(self, cap: cv2.VideoCapture) -> None:
		"""Process the video stream frame by frame."""
		while True:
			ret, frame = cap.read()
			if not ret:
				print("Failed to grab frame, ending loop.")
				break

			results = self._get_tracking_results(frame)
			annotated_frame = frame.copy()

			current_centers = {}
			current_track_ids = set()

			for result in results:
				boxes = result.boxes
				ids = getattr(boxes, 'id', None)
				if ids is None:
					continue
				for i, track_id_tensor in enumerate(ids):
					track_id = int(track_id_tensor.item())
					x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
					center_x = (x1 + x2) // 2
					center_y = (y1 + y2) // 2
					current_centers[track_id] = (center_x, center_y)
					current_track_ids.add(track_id)

					is_moving = self.movement_detector.is_object_moving(track_id, center_x, center_y)
					if is_moving:
						self.handler(track_id=track_id, is_moving=is_moving)  # Example angle, adjust as needed
						self.annotator.annotate_moving_object(annotated_frame, track_id, x1, y1, x2, y2)

			self.movement_detector.cleanup_stale_tracks(current_track_ids)
			self.movement_detector.update_centers(current_centers)

			if self.debug:
				cv2.imshow('YOLO Camera with Movement Detection', annotated_frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

	def _get_tracking_results(self, frame: np.ndarray) -> Any:
		"""Get YOLO tracking results for the frame."""
		return self.model.track(
			frame,
			imgsz=self.camera_config.yolo_image_size,
			stream=True,
			verbose=False,
			conf=self.camera_config.confidence_threshold,
			persist=True,
			classes=self.camera_config.person_class_id
		)

	def _cleanup_resources(self, cap: cv2.VideoCapture) -> None:
		"""Clean up camera and display resources."""
		print("Releasing resources.")
		cap.release()
		cv2.destroyAllWindows()
