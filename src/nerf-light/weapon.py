from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep

# TODO: Shooting system (hotwired nerf trigger)


class Weapon:
	def __init__(self, min_angle: int = 0, max_angle: int = 100, pin: int = 14):
		factory = PiGPIOFactory()
		self.trigger = AngularServo(
			pin=pin,  # GPIO pin (change as needed)
			min_angle=0,
			max_angle=180,
			min_pulse_width=0.5/1000,  # 0.5 ms
			max_pulse_width=2.5/1000,  # 2.5 ms
			pin_factory=factory
		)
		self.min_angle = min_angle
		self.max_angle = max_angle

	def fire(self):
		"""Fire the Nerf turret."""
		self.trigger.angle = self.min_angle
		sleep(0.1)
		print(self.max_angle)
		self.trigger.angle = self.max_angle
		sleep(1)
		self.trigger.angle = self.min_angle
		sleep(2)
