import os
import platform
import time
import board
import busio
import subprocess
from typing import Tuple
import smbus2 as smbus
import threading
import time
from time import perf_counter
import numpy as np
import simpleaudio as sa
import struct
from typing import List, Tuple




def is_raspberry_pi() -> bool:
    # Check platform name
    if platform.system() != "Linux":
        return False
    
    # Check /proc/cpuinfo for Raspberry Pi specific strings
    try:
        with open("/proc/cpuinfo", "r") as cpuinfo:
            for line in cpuinfo:
                if "Raspberry Pi" in line or "BCM" in line:
                    return True
    except FileNotFoundError:
        return False
    return False

def get_git_commit_hash() -> str:
    try:
        # Run git command to get the short commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
        return commit_hash
    except Exception as e:
        return f"Error retrieving Git commit hash: {e}"


class I2CController:
    def __init__(self, address: int = 0x08):
        self.bus = smbus.SMBus(1)  # Use 1 for newer Raspberry Pi versions
        self.address = address
        self.touch_data = [0, 0]  # Store data from both MPR121s
        self.current_bpm = 120
        self._lock = threading.Lock()
        
    def read_touch_data(self) -> List[List[int]]:
        """
        Read touch data from Arduino and convert to grid format.
        Returns a 4x20 list where 1 indicates a touched position.
        The grid is addressed as follows:
        - First MPR121: 12 columns (0-11)
        - Second MPR121: 8 columns (12-19) and 4 rows
        """
        try:
            with self._lock:
                # Read 5 bytes: 4 for touch data (2 per MPR121) + 1 for BPM
                data = self.bus.read_i2c_block_data(self.address, 0, 5)
                
                # Extract touch data
                touch2 = (data[1] << 8) | data[0]  # First MPR121 (12 columns)
                touch1 = (data[3] << 8) | data[2]  # Second MPR121 (8 columns + 4 rows)
                self.current_bpm = data[4]

                print("recieved: ", touch1, touch2, data[4])
                
                # Initialize grid
                grid = [[0 for _ in range(20)] for _ in range(4)]
                
                # Get active row(s) from second MPR121 (last 4 bits)
                active_rows = []

                #TODO: this can be cut short, by checking rows first

                bits1 = bin(touch1)[2:]
                bits2 = bin(touch2)[2:]

                cols = bits1+bits2[0:7]
                rows = bits2[7:]

                for c in cols:
                    if int(c) > 0:
                        grid[1][c] = 1                       

                print("====")
                for row in grid:
                    print(row,"\n")
                print("====")
                
                return grid
                
        except Exception as e:
            print(f"I2C read error: {e}")
            return [[0 for _ in range(20)] for _ in range(4)]
    
    def send_position(self, position: int):
        """
        Send current sequencer position to Arduino for LED display.
        track: 0-3
        position: 0-19
        """
        try:
            with self._lock:
                # Pack track and position into single byte
                data = (position & 0x3F)
                print("sending position: ", position, "and data:", data)
                self.bus.write_i2c_block_data(self.address, 0x01, [data])
        except Exception as e:
            print(f"I2C write error (position): {e}")
    
    def send_sample_state(self, track: int, position: int, active: bool):
        """
        Send sample state to Arduino for LED display.
        track: 0-3
        position: 0-19
        active: True if sample is active
        """
        try:
            with self._lock:
                # Pack track and position into single byte
                data = (track << 6) | (position & 0x3F)
                print("sending sample state:", [data, 1 if active else 0] )
                self.bus.write_i2c_block_data(self.address, 0x02, [data, 1 if active else 0])
        except Exception as e:
            print(f"I2C write error (sample state): {e}")

    def get_bpm(self) -> int:
        """Returns the current BPM value from the rotary encoder."""
        return self.current_bpm

def update_sequencer_from_touch(i2c: I2CController, sequencer_on: List[List[int]], sequencer_changed: List[int]):
    """
    Continuously update sequencer state based on touch input.
    """
    while True:
        grid = i2c.read_touch_data()
        
        # Update sequencer state based on touch data
        for row in range(4):
            for col in range(20):
                if grid[row][col]:  # If position is touched
                    # Toggle the state
                    sequencer_on[row][col] = 1 - sequencer_on[row][col]
                    sequencer_changed[col] = 1
                    # Send new state to Arduino
                    i2c.send_sample_state(row, col, sequencer_on[row][col] == 1)
        
        time.sleep(0.1)  # Small delay to prevent overwhelming the I2C bus


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

# Convert raw samples to 16-bit PCM format
def to_waveobject(samples):
    # Convert float samples (-1.0 to 1.0) to 16-bit PCM format (-32767 to 32767)
    samples = np.int16(samples * 32767)
    # Return the WaveObject created from the byte data
    return sa.WaveObject(samples.tobytes(), num_channels=1, bytes_per_sample=2, sample_rate=44100)

# Create an empty WaveObject for silence
def create_silent_wave():
    silent_samples = np.zeros(44100)  # 1 second of silence
    return to_waveobject(silent_samples)

def resample_audio(numpy_array, original_channels, target_channels=2):
    if original_channels == target_channels:
        return numpy_array
    elif original_channels == 1 and target_channels == 2:
        return np.column_stack((numpy_array, numpy_array))
    elif original_channels == 2 and target_channels == 1:
        return np.mean(numpy_array.reshape(-1, 2), axis=1)
    else:
        raise ValueError("Unsupported channel conversion")

def Numpy2Wave(numpy_array: np.ndarray) -> 'WaveObject':
    # Ensure the input is in the range [-1, 1]
    numpy_array = np.clip(numpy_array, -1, 1)
    
    # Ensure the array is 2D (stereo)
    if len(numpy_array.shape) == 1:
        numpy_array = np.column_stack((numpy_array, numpy_array))
    elif numpy_array.shape[1] == 1:
        numpy_array = np.repeat(numpy_array, 2, axis=1)
    
    # Convert to int32 (3 bytes per sample)
    max_value = 2**(24 - 1) - 1  # Max value for 24-bit audio
    numpy_array = (numpy_array * max_value).astype(np.int32)
    
    # Pack as 3-byte integers
    byte_data = b''.join(struct.pack('<i', val)[:3] for val in numpy_array.flatten())
    
    return sa.WaveObject(byte_data, num_channels=2, bytes_per_sample=3, sample_rate=44100)

def Wave2numpy(wave_obj):
    byte_data = wave_obj.audio_data
    num_samples = len(byte_data) // (wave_obj.num_channels * wave_obj.bytes_per_sample)
    
    if wave_obj.bytes_per_sample == 2:
        fmt = '<' + 'h' * (num_samples * wave_obj.num_channels)
        unpacked = struct.unpack(fmt, byte_data)
        numpy_array = np.array(unpacked, dtype=np.float32) / 32768.0
    elif wave_obj.bytes_per_sample == 3:
        # Correctly handle 3-byte integers
        numpy_array = np.zeros(num_samples * wave_obj.num_channels, dtype=np.float32)
        for i in range(0, len(byte_data), 3):
            value = int.from_bytes(byte_data[i:i+3], byteorder='little', signed=True)
            numpy_array[i//3] = value / 8388608.0  # Normalize to [-1, 1]
    else:
        raise ValueError("Unsupported bytes_per_sample")
    
    return numpy_array.reshape(-1, wave_obj.num_channels)


def render():
    """ Render the audio for columns that have changed. """
    global SEQUENCER_AUDIO, SEQUENCER_CHANGED, RAW_SAMPLES, SEQUENCER_ON

    while True:
        for col_index, changed in enumerate(SEQUENCER_CHANGED):
            if changed > 0:  # Only process columns where a change occurred
                # Convert selected RAW_SAMPLES to numpy arrays
                active_samples_indices = [row_index for row_index, is_active in enumerate(SEQUENCER_ON) if is_active[col_index] == 1]
                selected_samples = [RAW_SAMPLES[i] for i in active_samples_indices]
                
                if not selected_samples:
                    selected_samples = [np.zeros((44100, 2))]
                
                # Determine the maximum length
                max_length = max(sample.shape[0] for sample in selected_samples)
                
                # Create an array of zeros for stereo output
                mixed_samples = np.zeros((max_length, 2))
                
                # Sum the samples, handling shorter samples and different channel counts
                for sample in selected_samples:
                    # Ensure sample is 2D
                    if len(sample.shape) == 1:
                        sample = sample.reshape(-1, 1)
                    
                    # Convert mono to stereo if necessary
                    if sample.shape[1] == 1:
                        sample = np.repeat(sample, 2, axis=1)
                    
                    # Pad shorter samples with zeros to match max_length
                    padded_sample = np.pad(sample, ((0, max_length - sample.shape[0]), (0, 0)), 'constant')
                    
                    # Add the padded sample to mixed_samples
                    mixed_samples += padded_sample
                
                # Normalize the mixed_samples to prevent clipping
                if np.max(np.abs(mixed_samples)) > 1.0:
                    mixed_samples /= np.max(np.abs(mixed_samples))
                
                # Convert the mixed_samples to a WaveObject
                SEQUENCER_AUDIO[col_index] = Numpy2Wave(mixed_samples)
                
                # Reset the change flag for this column
                SEQUENCER_CHANGED[col_index] = 0
        
        # Add a small delay to prevent high CPU usage
        time.sleep(0.01)  # Sleep for 10 milliseconds to reduce CPU load

                    
def load_n_samples(folder_path, n):
    wav_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.wav')])[:n]
    samples = [sa.WaveObject.from_wave_file(os.path.join(folder_path, wav_file)) for wav_file in wav_files]

    samples_out = []

    for s in samples:
        print(f"Original: {s.num_channels} channels, {s.bytes_per_sample} bytes per sample, {s.sample_rate} Hz")
        numpy_array = Wave2numpy(s)
        
        samples_out.append(numpy_array)



    return samples_out


# Main loop with metronome and PID control
def main_loop(i2c: I2CController):
    global SEQUENCER_AUDIO, SEQUENCER_CHANGED, RAW_SAMPLES, SEQUENCER_GLOBAL_STEP, BPM, STOPED, SEQUENCER_ON

    # Load samples
    RAW_SAMPLES = load_n_samples("./", SEQUENCE_SAMPLES)

    # Use BPM from encoder
    bpm = 120#i2c.get_bpm()
    delay = d = wait_time = 60/bpm
    print(f'{60 / delay} bpm')

    a = perf_counter()
    calculated = True
    pid = PIDController(Kp=0.5, Ki=0.1, Kd=0.01)

    SEQUENCER_GLOBAL_STEP = 0
    SEQUENCER_ON = [[0 for _ in range(20)] for _ in range(4)]
    SEQUENCER_CHANGED = [0 for _ in range(20)]
    SEQUENCER_AUDIO = [create_silent_wave() for _ in range(20)]

    STOPED = False

    while not STOPED:
        # Update BPM from encoder
        new_bpm = i2c.get_bpm()
        if new_bpm != bpm:
            bpm = new_bpm
            d = 60/bpm
            print(f'New BPM: {bpm}')

        b = perf_counter()
        te = abs(b-a)

        if (te > wait_time) or a == 0:
            a = perf_counter()
            
            # Play audio
            SEQUENCER_AUDIO[SEQUENCER_GLOBAL_STEP].play()
            
            # Send current position to Arduino
            i2c.send_position(SEQUENCER_GLOBAL_STEP)

            # Update step
            SEQUENCER_GLOBAL_STEP = (SEQUENCER_GLOBAL_STEP + 1) % SEQUENCE_LENGTH
            
            delay = te - d
            calculated = False
            
        else:
            if not calculated:
                correction = pid.update(delay, d)
                wait_time = max(0, d - correction)
                calculated = True




#### Global/Shared variables ######
pressed_key = ""

SEQUENCE_LENGTH = 16
SEQUENCE_SAMPLES = 4

BPM = 120

STOPED = False

# Load audio files
#wave_obj = sa.WaveObject.from_wave_file("metronome.wav")  # Metronome sound

SEQUENCER_ON = [[0 for _ in range(SEQUENCE_LENGTH)] for _ in range(SEQUENCE_SAMPLES)]
RAW_SAMPLES = [0 for _ in range(SEQUENCE_SAMPLES)]
SEQUENCER_AUDIO = [create_silent_wave() for _ in range(SEQUENCE_LENGTH)] 
SEQUENCER_AUDIO_new = [create_silent_wave() for _ in range(SEQUENCE_LENGTH)] 

SEQUENCER_GLOBAL_STEP = 0

SEQUENCER_CHANGED = [0 for _ in range(SEQUENCE_LENGTH)] 


def main():
    print(f"Running commit: {get_git_commit_hash()}")
    
    if not is_raspberry_pi():
        print("Not running on a Raspberry Pi.")
        return
    
    print("Running on a Raspberry Pi.")

    i2c = I2CController()

    # Start the audio rendering thread
    sound_thread = threading.Thread(target=render)
    sound_thread.daemon = True
    sound_thread.start()

    # Start the touch input thread
    touch_thread = threading.Thread(target=update_sequencer_from_touch, 
                                  args=(i2c, SEQUENCER_ON, SEQUENCER_CHANGED))
    touch_thread.daemon = True
    touch_thread.start()

    # Start the main loop
    main_loop(i2c)

if __name__ == "__main__":
    main()