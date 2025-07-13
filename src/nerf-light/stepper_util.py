import time
from gpiozero import OutputDevice


class StepperMotor:
	def __init__(self, dir_pin: int, step_pin: int):
		self.dir_pin = OutputDevice(dir_pin)
		self.step_pin = OutputDevice(step_pin)
		self.dir_pin.on()
		self.step_angle: float = 1.8
		self.weapon_angle = 0

	def move_steps(self, steps: int, delay: float = 0.001):
		"""Move the stepper motor a specified number of steps."""

		if steps < 0:
			self.dir_pin.off()
		else:
			self.dir_pin.on()

		for _ in range(abs(steps)):
			self.step_pin.on()
			time.sleep(delay)
			self.step_pin.off()
			time.sleep(delay)

	def move_degrees(self, degrees: float, delay: float = 0.001):
		"""Move the stepper motor a specified number of degrees."""
		self.weapon_angle += degrees
		self.weapon_angle %= 360

		steps = int(degrees // self.step_angle)
		self.move_steps(steps, delay)

	def set_angle(self, angle: float, delay: float = 0.001):
		"""Move the stepper motor to a specified angle."""
		angle %= 360
		diff = (angle - self.weapon_angle + 540) % 360 - 180
		self.move_degrees(diff, delay)

		return self.weapon_angle
