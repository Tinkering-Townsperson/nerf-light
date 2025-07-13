import RPi.GPIO as GPIO
import time

DIR = 2    # GPIO pin 2
STEP = 3   # GPIO pin 3

GPIO.setmode(GPIO.BCM)
GPIO.setup(DIR, GPIO.OUT)
GPIO.setup(STEP, GPIO.OUT)

GPIO.output(DIR, GPIO.HIGH)  # Set direction

# Move 200 steps (1 rotation if 1.8° step angle)
for _ in range(200):
    GPIO.output(STEP, GPIO.HIGH)
    time.sleep(0.001)  # Step pulse width
    GPIO.output(STEP, GPIO.LOW)
    time.sleep(0.001)

GPIO.cleanup()
