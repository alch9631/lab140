from gpiozero import Motor, Button
from time import sleep

# Motor-Pins anpassen
left_motor = Motor(forward=23, backward=24)
right_motor = Motor(forward=27, backward=22)

# Encoder-Pins anpassen
left_encoder = Button(17)
right_encoder = Button(18)

# Encoder-Zähler
left_count = 0
right_count = 0

def left_pulse():
    global left_count
    left_count += 1

def right_pulse():
    global right_count
    right_count += 1

left_encoder.when_pressed = left_pulse
right_encoder.when_pressed = right_pulse

# Roboter-Parameter
wheel_diameter = 0.1  # Meter
wheel_circumference = 3.1416 * wheel_diameter
encoder_counts_per_rev = 360
distance_to_move = 1.0

target_counts = int((distance_to_move / wheel_circumference) * encoder_counts_per_rev)
print(f"Target counts: {target_counts}")

# Reset counters
left_count = 0
right_count = 0

# Motor starten
left_motor.forward()
right_motor.forward()

try:
    while left_count < target_counts or right_count < target_counts:
        # optional: einfacher Geradeaus-Abgleich
        if left_count > right_count:
            left_motor.stop()
        elif right_count > left_count:
            right_motor.stop()
finally:
    left_motor.stop()
    right_motor.stop()
    print(f"Final counts: Left={left_count}, Right={right_count}")
    print("Robot moved 1 meter!")