import time
from gpiozero import OutputDevice

DIR = 2    # GPIO pin 2
STEP = 3   # GPIO pin 3

dir_pin = OutputDevice(DIR)
step_pin = OutputDevice(STEP)

dir_pin.on()  # Set direction (HIGH)

# Move 200 steps (1 rotation if 1.8° step angle)
for _ in range(200):
	step_pin.on()
	time.sleep(0.001)  # Step pulse width
	step_pin.off()
	time.sleep(0.001)


class StepperMotor:
	def __init__(self, dir_pin: int, step_pin: int):
		self.dir_pin = OutputDevice(dir_pin)
		self.step_pin = OutputDevice(step_pin)
		self.dir_pin.on()
		self.step_angle: float = 1.8

	def move_steps(self, steps: int, delay: float = 0.001):
		"""Move the stepper motor a specified number of steps."""

		if steps < 0:
			self.dir_pin.off()
		else:
			self.dir_pin.on()

		for _ in range(steps):
			step_pin.on()
			time.sleep(delay)
			step_pin.off()
			time.sleep(delay)

	def move_degrees(self, degrees: float, delay: float = 0.001):
		"""Move the stepper motor a specified number of degrees."""
		steps = int(degrees / self.step_angle)
		self.move_steps(steps, delay)
