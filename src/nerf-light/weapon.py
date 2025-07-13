from .stepper_util import StepperMotor

from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep  # noqa

# TODO: Shooting system (hotwired nerf trigger)


class Weapon:
	def __init__(self, min_angle: int = 0, max_angle: int = 100):
		factory = PiGPIOFactory()
		self.trigger = AngularServo(
			14,  # GPIO pin (change as needed)
			min_angle=0,
			max_angle=180,
			min_pulse_width=0.5/1000,  # 0.5 ms
			max_pulse_width=2.5/1000,  # 2.5 ms
			pin_factory=factory
		)
		self.stepper = StepperMotor(2, 3)
		self.min_angle = min_angle
		self.max_angle = max_angle

	def aim(self, angle: int):
		"""Aim the Nerf turret."""
		self.stepper.set_angle(angle)

	def fire(self):
		"""Fire the Nerf turret."""
		self.trigger.angle = self.min_angle
		sleep(0.1)
		print(self.max_angle)
		self.trigger.angle = self.max_angle
		sleep(1)
		self.trigger.angle = self.min_angle
		sleep(2)
