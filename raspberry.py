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
    # MPR121 Register addresses
    TOUCH_STATUS_REG = 0x00
    ELE_CFG_REG = 0x5E
    MHD_R = 0x2B
    NHD_R = 0x2C
    NCL_R = 0x2D
    FDL_R = 0x2E
    MHD_F = 0x2F
    NHD_F = 0x30
    NCL_F = 0x31
    FDL_F = 0x32
    NHDT = 0x33
    NCLT = 0x34
    FDLT = 0x35
    MHDF = 0x36
    NHDF = 0x37
    NCLF = 0x38
    FDLF = 0x39
    ELE0_T = 0x41
    ELE0_R = 0x42
    AUTO_CONFIG_0 = 0x7B
    AUTO_CONFIG_1 = 0x7C

    def __init__(self, arduino_address: int = 0x08, bus_number: int = 1):
        """
        Initialize the I2C controller with both Arduino and MPR121 support.
        
        Args:
            arduino_address: I2C address of the Arduino
            bus_number: I2C bus number (usually 1 for newer Raspberry Pi)
        """
        self.bus = smbus.SMBus(bus_number)
        self.arduino_address = arduino_address
        self.mpr121_address1 = 0x5A
        self.mpr121_address2 = 0x5B
        self.touch_data = [0, 0]  # Store data from both MPR121s
        self.current_bpm = 120
        self._lock = threading.Lock()

        # Initialize both MPR121 sensors
        for address in [self.mpr121_address1, self.mpr121_address2]:
            self._init_mpr121(address)

    def _init_mpr121(self, address):
        """Initialize a single MPR121 sensor with default configuration."""
        try:
            with self._lock:
                # Soft reset
                self.bus.write_byte_data(address, self.ELE_CFG_REG, 0x00)
                
                # Configure touch and release thresholds for all electrodes
                for i in range(12):
                    self.bus.write_byte_data(address, self.ELE0_T + i*2, 12)  # Touch threshold
                    self.bus.write_byte_data(address, self.ELE0_R + i*2, 6)   # Release threshold
                
                # Configure baseline filtering
                self.bus.write_byte_data(address, self.MHD_R, 0x01)
                self.bus.write_byte_data(address, self.NHD_R, 0x01)
                self.bus.write_byte_data(address, self.NCL_R, 0x00)
                self.bus.write_byte_data(address, self.FDL_R, 0x00)
                
                # Enable all electrodes and auto configuration
                self.bus.write_byte_data(address, self.ELE_CFG_REG, 0x0C)
                self.bus.write_byte_data(address, self.AUTO_CONFIG_0, 0x0B)
                self.bus.write_byte_data(address, self.AUTO_CONFIG_1, 0x9F)
                
        except IOError as e:
            print(f"Error initializing MPR121 at address 0x{address:02X}: {e}")
            raise

    def read_touch_sensors(self) -> Tuple[list, list]:
        """Read the status of all inputs from both MPR121 sensors."""
        try:
            with self._lock:
                # Read touch status registers (2 bytes each sensor)
                status1 = self.bus.read_word_data(self.mpr121_address1, self.TOUCH_STATUS_REG)
                status2 = self.bus.read_word_data(self.mpr121_address2, self.TOUCH_STATUS_REG)
                
            # Convert to list of boolean values for each electrode
            sensor1_status = [(status1 & (1 << i)) != 0 for i in range(12)]
            sensor2_status = [(status2 & (1 << i)) != 0 for i in range(12)]
            
            return sensor1_status, sensor2_status
            
        except IOError as e:
            print(f"Error reading touch sensors: {e}")
            return [False] * 12, [False] * 12

    def print_touched_inputs(self):
        """Print which inputs are currently being touched on both sensors."""
        sensor1_status, sensor2_status = self.read_touch_sensors()
        
        # Check sensor 1
        for i, touched in enumerate(sensor1_status):
            if touched:
                print(f"Sensor 1 - Input {i} is touched.")
                
        # Check sensor 2
        for i, touched in enumerate(sensor2_status):
            if touched:
                print(f"Sensor 2 - Input {i} is touched.")

    def send_position(self, position: int):
        """
        Send current sequencer position to Arduino for LED display.
        
        Args:
            position: 0-63 position value
        """
        try:
            with self._lock:
                data = (position & 0x3F)  # Ensure position is within 6-bit range
                print(f"sending position: {position} and data: {data}")
                self.bus.write_i2c_block_data(self.arduino_address, 0x01, [data])

        except Exception as e:
            print(f"I2C write error (position): {e}")


    def get_bpm(self) -> int:
        """Returns the current BPM value from the rotary encoder."""
        try:
            with self._lock:
                # Read 4 bytes to receive the full `long` value
                data = self.bus.read_i2c_block_data(self.arduino_address, 0, 4)
                
                # Convert 4 bytes to a long integer
                print(bytes(data))
                self.current_bpm = struct.unpack('<L', bytes(data))[0]  # '<l' is for little-endian long
                
                print("Received BPM:", self.current_bpm)
                return self.current_bpm
        except Exception as e:
            print(f"I2C write error (position): {e}")

        return self.current_bpm

    def __del__(self):
        """Clean up the SMBus connection when the object is destroyed."""
        try:
            self.bus.close()
        except:
            pass
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
        

        b = perf_counter()
        te = abs(b-a)

        if (te > wait_time) or a == 0:
            a = perf_counter()
            
            # Play audio
            SEQUENCER_AUDIO[SEQUENCER_GLOBAL_STEP].play()
            
            # Update step
            SEQUENCER_GLOBAL_STEP = (SEQUENCER_GLOBAL_STEP + 1) % SEQUENCE_LENGTH
            
            delay = te - d
            calculated = False
            
        else:
            if not calculated:

                
                
                # Send current position to Arduino
                i2c.send_position(SEQUENCER_GLOBAL_STEP)
                i2c.print_touched_inputs()                
                correction = pid.update(delay, d)
                wait_time = max(0, d - correction)
                calculated = True
                
            # TODO: DO LIKE A MODULO HERE TO UPDATE STUFF (every 50 th wait or smth)
            #         - like update bpm
            #         - like recieve touch update, send 8 Byte display update
            # Update BPM from encoder
                new_bpm = i2c.get_bpm()
                if new_bpm != bpm:
                    bpm = new_bpm
                    d = 60/120#bpm
                    print(f'New BPM: {bpm}')



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

    # Start the main loop
    main_loop(i2c)

if __name__ == "__main__":
    main()