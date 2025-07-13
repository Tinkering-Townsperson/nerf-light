__version__ = "v0.1.0"

# from . import camera_util
from . import weapon

from os import getenv
from time import sleep


def main() -> None:
	debug = True if getenv("DEBUG") else True  # noqa

	w = weapon.Weapon(min_angle=0, max_angle=90)
	w.aim(45)
	sleep(1)
	w.aim(-45)
	sleep(1)
	w.aim(90)
	sleep(1)
	w.aim(-90)

	# camera_source = int(getenv("CAMERA_STREAM", "0"))
	# model_path = getenv("YOLO_MODEL_PATH", "yolo11n.pt")

	# camera = camera_util.Camera(model_path=model_path, source=camera_source, debug=debug)
	# camera.mainloop()
