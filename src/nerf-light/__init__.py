__version__ = "v0.1.0"

from os import getenv
from dataclasses import dataclass


@dataclass
class Config(object):
	DEBUG: bool = True if getenv("DEBUG") else False
	CAMERA_SOURCE: int = int(getenv("CAMERA_STREAM", "0"))
	MODEL_PATH: str = getenv("YOLO_MODEL_PATH", "yolo11n.pt")
	SERVO_PIN: int = int(getenv("SERVO_PIN", "14"))
	RED_LED_PIN: int = int(getenv("RED_LED_PIN", "24"))
	GREEN_LED_PIN: int = int(getenv("GREEN_LED_PIN", "23"))


def main():
	from .game import Game

	game = Game()
	game.start()
