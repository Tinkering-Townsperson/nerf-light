from gpiozero import AngularServo, OutputDevice
from gpiozero.pins.pigpio import PiGPIOFactory
import time
from typing import Tuple, Optional


class PID:
    """A simple PID controller."""

    def __init__(self, Kp: float, Ki: float, Kd: float, setpoint: float = 0):
        """
        Initialize the PID controller.

        Args:
            Kp: Proportional gain.
            Ki: Integral gain.
            Kd: Derivative gain.
            setpoint: The desired value to maintain.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        
        self._previous_error = 0.0
        self._integral = 0.0

    def update(self, current_value: float) -> float:
        """
        Calculate the PID output.

        Args:
            current_value: The current measured value.

        Returns:
            The PID output signal.
        """
        error = self.setpoint - current_value
        self._integral += error
        derivative = error - self._previous_error
        
        output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)
        
        self._previous_error = error
        return output

    def reset(self) -> None:
        """Resets the integral and previous error."""
        self._integral = 0.0
        self._previous_error = 0.0


class TurretController:
    """Controls the aiming of the weapon turret, assuming a stationary camera."""

    def __init__(self, stepper_dir_pin: int, stepper_step_pin: int, pid_controller: PID, steps_per_pixel: float, aim_tolerance: int, camera_width: int):
        """
        Initializes the turret controller.

        Args:
            stepper_dir_pin: GPIO pin for the stepper motor direction.
            stepper_step_pin: GPIO pin for the stepper motor step.
            pid_controller: The PID controller instance for aiming.
            steps_per_pixel: Calibration factor for converting pixel distance to motor steps.
            aim_tolerance: The error tolerance in pixels to be considered "on target".
            camera_width: The width of the camera frame in pixels.
        """
        self.factory = PiGPIOFactory()
        self.stepper_dir = OutputDevice(stepper_dir_pin, pin_factory=self.factory)
        self.stepper_step = OutputDevice(stepper_step_pin, pin_factory=self.factory)
        self.pid_controller = pid_controller
        self.steps_per_pixel = steps_per_pixel  # e.g., 0.5 steps per pixel
        self.pixels_per_step = 1 / steps_per_pixel if steps_per_pixel != 0 else 0
        self.aim_tolerance = aim_tolerance
        self.camera_width = camera_width
        
        # State variable to track the turret's current aiming position in pixels.
        # We assume it starts centered.
        self.current_aim_x = camera_width / 2
        
        self.step_delay = 0.001  # Controls max speed
        self.last_error = 0.0

    def update(self, target_x: Optional[float]) -> None:
        """
        Updates the turret's aim to track the target's position.

        Args:
            target_x: The horizontal coordinate of the target. If None, the turret stops.
        """
        if target_x is None:
            self.pid_controller.reset()
            # No target, so error is considered zero.
            self.last_error = 0
            return

        # The goal is to move our aim (current_aim_x) to the target's position (target_x).
        self.pid_controller.setpoint = target_x
        
        # Calculate PID output based on our current aiming position.
        # The output is the required adjustment.
        pid_output = self.pid_controller.update(self.current_aim_x)
        
        # Convert the required adjustment (in pixels) to motor steps.
        steps_to_move = int(pid_output * self.steps_per_pixel)
        
        # Move the motor and update our internal state of where we are aiming.
        if self._move_stepper(steps_to_move):
            # Update our aim position based on how many steps we actually moved.
            self.current_aim_x += steps_to_move * self.pixels_per_step

        # Update the error for external checks.
        self.last_error = self.current_aim_x - target_x

    def is_on_target(self) -> bool:
        """Checks if the turret is aimed at the target within tolerance."""
        return abs(self.last_error) < self.aim_tolerance

    def _move_stepper(self, steps: int) -> bool:
        """
        Moves the stepper motor a specific number of steps.
        Returns True if movement occurred.
        """
        if steps == 0:
            return False

        # Set direction
        self.stepper_dir.on() if steps > 0 else self.stepper_dir.off()

        # Perform steps
        for _ in range(abs(steps)):
            self.stepper_step.on()
            time.sleep(self.step_delay)
            self.stepper_step.off()
            time.sleep(self.step_delay)
        
        return True

    def cleanup(self) -> None:
        """Cleans up GPIO resources."""
        self.stepper_dir.close()
        self.stepper_step.close()


class WeaponController:
    """Manages all weapon systems, including aiming and firing."""

    def __init__(self, trigger_pin: int = 14, stepper_dir_pin: int = 2, stepper_step_pin: int = 3, pid_controller: Optional[PID] = None, steps_per_pixel: float = 0.5, aim_tolerance: int = 10, camera_width: int = 640):
        """
        Initializes the weapon controller.

        Args:
            trigger_pin: GPIO pin for the trigger servo.
            stepper_dir_pin: GPIO pin for the stepper motor direction.
            stepper_step_pin: GPIO pin for the stepper motor step.
            pid_controller: An instance of the PID controller for aiming.
            steps_per_pixel: Calibration factor for aiming.
            aim_tolerance: Pixel tolerance for being on target.
            camera_width: The width of the camera frame in pixels.
        """
        self.factory = PiGPIOFactory()
        
        # Configure Trigger Servo
        self.trigger = AngularServo(
            trigger_pin,
            min_angle=0,
            max_angle=45,  # Restrict angle for safety
            min_pulse_width=0.5 / 1000,
            max_pulse_width=2.5 / 1000,
            pin_factory=self.factory
        )
        
        # Configure Turret Controller for aiming
        if pid_controller is None:
            raise ValueError("A PID controller must be provided to WeaponController.")
            
        self.turret = TurretController(
            stepper_dir_pin=stepper_dir_pin,
            stepper_step_pin=stepper_step_pin,
            pid_controller=pid_controller,
            steps_per_pixel=steps_per_pixel,
            aim_tolerance=aim_tolerance,
            camera_width=camera_width
        )

        # Weapon state
        self.is_firing = False
        self.reset()

    def reset(self) -> None:
        """Resets the trigger to the safe position."""
        self.trigger.angle = 0
        self.is_firing = False
        print("Weapon reset and ready.")

    def fire(self, duration: float = 0.5) -> None:
        """
        Activates the trigger servo to fire the weapon.

        Args:
            duration: Time in seconds to hold the trigger.
        """
        if not self.is_firing:
            print("FIRING!")
            self.is_firing = True
            self.trigger.angle = 45  # Firing angle
            time.sleep(duration)
            self.reset() # Reset after firing

    def aim_at(self, target_x: Optional[float]) -> None:
        """
        Aims the turret at the given horizontal coordinate.

        Args:
            target_x: The target's x-coordinate. If None, aiming stops.
        """
        self.turret.update(target_x)

    def is_aimed(self) -> bool:
        """Checks if the weapon is currently aimed at the target."""
        return self.turret.is_on_target()

    def cleanup(self) -> None:
        """Cleans up GPIO resources."""
        self.turret.cleanup()
        self.trigger.close()
        print("Weapon systems cleaned up.")
