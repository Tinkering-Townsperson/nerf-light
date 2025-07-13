
# TODO: Servo/stepper motors for the turret
# TODO: Shooting system (hotwired nerf trigger)

from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep


factory = PiGPIOFactory()

trigger = AngularServo(
    14,  # GPIO pin (change as needed)
    min_angle=0,
    max_angle=180,
    min_pulse_width=0.5/1000,  # 0.5 ms
    max_pulse_width=2.5/1000,  # 2.5 ms
    pin_factory=factory
)

def fire():
    