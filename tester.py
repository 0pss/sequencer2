import numpy as np
from time import perf_counter
import simpleaudio as sa  # Assuming you're using simpleaudio for audio

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output

# Load your audio file
wave_obj = sa.WaveObject.from_wave_file("metronome.wav")  # Replace with your audio file

target_delay = 0.5  # 120 BPM
print(f"{60 / target_delay:.2f} bpm")

# Initialize PID controller
pid = PIDController(Kp=0.9, Ki=0.1, Kd=0.01)  # You may need to tune these values

accumulated_error = 0
start_time = perf_counter()

for i in range(100):
    beat_start = perf_counter()
    
    # Play the click
    play_obj = wave_obj.play()
    
    # Calculate the error
    elapsed = beat_start - start_time
    expected = i * target_delay
    error = elapsed - expected
    
    # Update PID controller
    correction = pid.update(error, target_delay)
    
    # Apply correction to wait time
    wait_time = max(0, target_delay - correction)
    
    # Wait until next beat
    while perf_counter() - beat_start < wait_time:
        pass
    
    actual_interval = perf_counter() - beat_start
    accumulated_error += abs(actual_interval - target_delay)
    
    print(f"Beat {i+1}: Interval = {actual_interval}s, Error = {error}s correction_factor = {wait_time-0.5}")

end_time = perf_counter()
total_time = end_time - start_time
print(f"\nTotal time: {total_time:.2f}s")
print(f"Accumulated error: {accumulated_error:.6f}s")
print(f"Average BPM: {(60 * 200 / total_time):.2f}")