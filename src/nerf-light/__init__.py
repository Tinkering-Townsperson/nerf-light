__version__ = "v0.1.0"

from . import camera_util

from os import getenv

from ultralytics import YOLO


def main() -> None:
	model_path = getenv("MODEL_PATH", "yolo11n.pt")

	debug = True if getenv("DEBUG") else False

	camera_source = int(getenv("CAMERA_STREAM", "0"))
	camera = camera_util.Camera(model_path=model_path, source=camera_source, debug=debug)
	camera.mainloop()
	camera.mainloop()
