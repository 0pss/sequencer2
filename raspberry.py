import os
import platform
import time
import board
import asyncio
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

class AsyncI2CController:
    def __init__(self, address: int = 0x08):
        self.bus = smbus.SMBus(1)
        self.address = address
        self.touch_data = [0, 0]
        self.current_bpm = 120
        self._lock = asyncio.Lock()
        
    async def read_touch_data(self) -> List[List[int]]:
        """
        Read touch data from Arduino and convert to grid format.
        Returns a 4x20 list where 1 indicates a touched position.
        """
        try:
            async with self._lock:
                # Read 5 bytes: 4 for touch data (2 per MPR121) + 1 for BPM
                data = self.bus.read_i2c_block_data(self.address, 0, 5)
                
                touch2 = (data[1] << 8) | data[0]
                touch1 = (data[3] << 8) | data[2]
                self.current_bpm = data[4]

                grid = [[0 for _ in range(20)] for _ in range(4)]

                bits1 = bin(touch1)[2:].zfill(12)
                bits2 = bin(touch2)[2:].zfill(12)
                print("received: ", bits1, bits2, data[4])

                cols = bits1 + bits2[:7]
                rows = bits2[8:]

                for i, c in enumerate(cols):
                    if int(c) > 0:
                        grid[0][i] = 1

                return grid
                
        except Exception as e:
            print(f"I2C read error: {e}")
            return [[0 for _ in range(20)] for _ in range(4)]
    
    async def send_position(self, position: int):
        """Send current sequencer position to Arduino for LED display."""
        try:
            async with self._lock:
                data = (position & 0x3F)
                print("sending position: ", position, "and data:", data)
                self.bus.write_i2c_block_data(self.address, 0x01, [data])
        except Exception as e:
            print(f"I2C write error (position): {e}")
    
    async def send_sample_state(self, track: int, position: int, active: bool):
        """Send sample state to Arduino for LED display."""
        try:
            async with self._lock:
                data = (track << 6) | (position & 0x3F)
                self.bus.write_i2c_block_data(self.address, 0x02, [data, 1 if active else 0])
        except Exception as e:
            print(f"I2C write error (sample state): {e}")

    def get_bpm(self) -> int:
        return self.current_bpm

async def update_sequencer_from_touch(i2c: AsyncI2CController, 
                                    sequencer_on: List[List[int]], 
                                    sequencer_changed: List[int]):
    """Continuously update sequencer state based on touch input."""
    while True:
        grid = await i2c.read_touch_data()
        
        # Update sequencer state based on touch data
        for row in range(4):
            for col in range(20):
                if grid[row][col]:
                    sequencer_on[row][col] = 1 - sequencer_on[row][col]
                    sequencer_changed[col] = 1
                    await i2c.send_sample_state(row, col, sequencer_on[row][col] == 1)
        
        # Use asyncio.sleep instead of time.sleep
        await asyncio.sleep(0.05)


async def main_loop(i2c: AsyncI2CController):
    """Main sequencer loop with metronome and PID control"""
    # Initialize variables
    bpm = 120
    delay = wait_time = 60/120#bpm
    pid = PIDController(Kp=0.5, Ki=0.1, Kd=0.01)
    
    SEQUENCER_GLOBAL_STEP = 0
    SEQUENCER_ON = [[0 for _ in range(20)] for _ in range(4)]
    SEQUENCER_CHANGED = [0 for _ in range(20)]
    SEQUENCER_AUDIO = [create_silent_wave() for _ in range(20)]

    last_tick = perf_counter()
    calculated = True

    while True:
        # Update BPM from encoder
        new_bpm = i2c.get_bpm()
        if new_bpm != bpm:
            bpm = new_bpm
            delay = 60/bpm
            print(f'New BPM: {bpm}')

        current_time = perf_counter()
        time_elapsed = current_time - last_tick

        if (time_elapsed > wait_time) or last_tick == 0:
            last_tick = current_time
            
            # Play audio
            SEQUENCER_AUDIO[SEQUENCER_GLOBAL_STEP].play()
            
            # Update step
            SEQUENCER_GLOBAL_STEP = (SEQUENCER_GLOBAL_STEP + 1) % SEQUENCE_LENGTH
            
            # Calculate timing error
            timing_error = time_elapsed - delay
            calculated = False
            
        else:
            if not calculated:
                # Send current position to Arduino
                await i2c.send_position(SEQUENCER_GLOBAL_STEP)
                
                # Update PID controller
                correction = pid.update(timing_error, delay)
                wait_time = max(0, delay - correction)
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


async def main():
    print(f"Running commit: {get_git_commit_hash()}")
    
    if not is_raspberry_pi():
        print("Not running on a Raspberry Pi.")
        return
    
    print("Running on a Raspberry Pi.")

    i2c = AsyncI2CController()

    # Create tasks for both the sequencer and touch input
    touch_task = asyncio.create_task(
        update_sequencer_from_touch(
            i2c, 
            SEQUENCER_ON, 
            SEQUENCER_CHANGED
        )
    )
    
    main_task = asyncio.create_task(main_loop(i2c))

    # Wait for both tasks
    await asyncio.gather(touch_task, main_task)

if __name__ == "__main__":
    asyncio.run(main())