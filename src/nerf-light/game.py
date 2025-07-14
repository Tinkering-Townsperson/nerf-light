from enum import Enum
from random import randint
from threading import Thread
from time import sleep

from gpiozero import LED

from . import Config
from .camera_util import Camera
from .weapon import Weapon


class GameState(Enum):
	PLAYING = 0
	WIN = 1
	LOSE = 2


class Game:
	def __init__(self):
		self.PAUSED = True
		self.state = GameState.PLAYING

		self.weapon: Weapon = Weapon(min_angle=0, max_angle=45, pin=Config.SERVO_PIN)

		self.camera: Camera = Camera(
			model_path=Config.MODEL_PATH,
			source=Config.CAMERA_SOURCE,
			debug=Config.DEBUG,
			weapon=self.weapon,
			handler=self.handle_movement
		)

		self.red_led = LED(Config.RED_LED_PIN)
		self.green_led = LED(Config.GREEN_LED_PIN)

	def start(self):
		self.camera_thread = Thread(target=self.camera.mainloop)
		self.camera_thread.start()

		self.game_thread = Thread(target=self.gameloop)
		self.game_thread.start()

	def gameloop(self):
		while self.state is GameState.PLAYING:
			print("green light")
			self.green_led.on()
			self.PAUSED = True
			sleep(randint(15, 45))
			self.green_led.off()
			# RED LIGHT
			sleep(1)
			print("red light")
			self.red_led.on()
			self.PAUSED = False
			sleep(10)
			self.red_led.off()

	def game_over(self):
		self.PAUSED = True
		self.camera_thread.join()
		self.game_thread.join()

		if self.state is GameState.WIN:
			print("You win!")
		elif self.state is GameState.LOSE:
			print("You lose!")

		self.red_led.off()
		self.green_led.off()

		exit(0)

	def handle_movement(self, track_id: int, is_moving: bool):
		"""Handle movement detection and aim the weapon."""
		if self.PAUSED:
			return

		if is_moving:
			print(f"Object #{track_id} was caught lacking!")
			self.weapon.fire()
			self.state = GameState.LOSE
