from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep


factory = PiGPIOFactory()

servo = AngularServo(
    14,  # GPIO pin (change as needed)
    pin_factory=factory
)

# Example: Move servo to min and max angles
try:
    while True:
        servo.angle = servo.min_angle  # Move to minimum angle
        sleep(1)
        servo.angle = servo.max_angle  # Move to maximum angle
        sleep(1)
except KeyboardInterrupt:
    servo.detach()  # This stops sending PWM signals
    print("Servo detached - free to move")
finally:
    servo.close()  # Ensure proper cleanup
