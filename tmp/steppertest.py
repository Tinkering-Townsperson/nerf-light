from gpiozero import OutputDevice
import time

DIR = OutputDevice(2)
STEP = OutputDevice(3)

try:
    # Spin one way for 3 seconds
    DIR.on()
    start_time = time.time()
    while time.time() - start_time < 3:
        STEP.on()
        time.sleep(0.001)
        STEP.off()
        time.sleep(0.001)

    # Reverse direction for 3 seconds
    DIR.off()
    start_time = time.time()
    while time.time() - start_time < 3:
        STEP.on()
        time.sleep(0.001)
        STEP.off()
        time.sleep(0.001)
finally:
    DIR.close()
    STEP.close()
