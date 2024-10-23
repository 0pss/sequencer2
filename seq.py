import numpy as np
import simpleaudio as sa
import threading
import os
import time
from time import perf_counter
from pynput import keyboard
import numpy as np
from time import perf_counter
import struct
import simpleaudio as sa  # Assuming you're using simpleaudio for audio



#######################################################
#             _______ ____  _____   ____              #
#            |__   __/ __ \|  __ \ / __ \             #
#               | | | |  | | |  | | |  | |            #
#               | | | |  | | |  | | |  | |            #
#               | | | |__| | |__| | |__| |            #
#               |_|  \____/|_____/ \____/             #
#                                                     #
# - Clean up unused stuff & break up into files       #
# - make render() function on demand, not cyclic      #
# - Implement Diagnostics                             #
# - Implement Doubletap (resampling)                  #
#     o Ist das äquivqlent zu doppelt so schnell?     #
# - Implement Live play (in former render() thread)   #
# - Implement setting.txt to load/save configs        #
# - Webserver to create configs ????                  #
# - Implement external trigger (raspi)                #
# - implement MPR121 input (raspi)                    #
# - Implement LED output (raspi)                      #
#                                                     #
#######################################################                 
                              


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



#### Global/Shared variables ######
pressed_key = ""

SEQUENCE_LENGTH = 8
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


# Generate a flexible key map based on available keys and sequence dimensions
# You can expand or adjust the key_pool to suit your needs
key_pool = (
    "12345678"  # Number row
    "qwertzui"  # First letter row
    "asdfghjk"   # Second letter row
    "yxcvbnm,"     # Third letter row
)

def create_dynamic_key_map():
    """ Dynamically map keys from the key pool to the sequencer. """
    key_map = {}
    index = 0
    for row in range(SEQUENCE_SAMPLES):
        for col in range(SEQUENCE_LENGTH):
            if index < len(key_pool):
                key_map[key_pool[index]] = (row, col)
                index += 1
            else:
                break
    print(key_map)
    return key_map

key_mappings = create_dynamic_key_map()

### /end global variables #########

def on_press(key):
    global pressed_key, STOPED
    try:
        pressed_key = key.char  # For regular keys
        if pressed_key in key_mappings:
            row, col = key_mappings[pressed_key]
            SEQUENCER_ON[row][col] = 1 - SEQUENCER_ON[row][col]  # Toggle the value (0 -> 1, 1 -> 0)
            SEQUENCER_CHANGED[col] = 1  # Mark that the column has changed
            #print(f"Key pressed: {pressed_key}, Updated SEQUENCER_ON[{row}][{col}] to {SEQUENCER_ON[row][col]}")
            print("------------------")
            for row in range(0,SEQUENCE_SAMPLES):
                print(SEQUENCER_ON[row])
            print("------------------")
        else:
            print(f"Key {pressed_key} not mapped.")
    except AttributeError:
        pressed_key = str(key)  # For special keys
        print(f"Special Key pressed: {pressed_key}")
        if pressed_key == 'Key.esc':
            STOPED = True


def on_release(key):
    if key == keyboard.Key.esc:  # Escape key to stop
        return False

def listen_for_keypress():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def render():
    """ Render the audio for columns that have changed. """
    global SEQUENCER_AUDIO, SEQUENCER_CHANGED

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
def main_loop():
    global SEQUENCER_AUDIO, SEQUENCER_CHANGED, RAW_SAMPLES, SEQUENCER_GLOBAL_STEP

    # First, load samples:
    RAW_SAMPLES = load_n_samples("./", SEQUENCE_SAMPLES)

    #input BPM
    bpm_input = input("Enter BPM (default 120): ") or "120"
    bpm = int(bpm_input) if bpm_input.isdigit() else 120
    delay = d = wait_time = 60/bpm
    print(60 / delay, 'bpm')

    # start the main loop
    a = perf_counter()
    calculated = True
    pid = PIDController(Kp=0.5, Ki=0.1, Kd=0.01)  # You may need to tune these values

    SEQUENCER_GLOBAL_STEP = 0

    while not STOPED:

        b = perf_counter()

        te = np.abs(b-a)

        if (te > wait_time) or a == 0:
            a=perf_counter()
            
            SEQUENCER_AUDIO[SEQUENCER_GLOBAL_STEP].play()

            SEQUENCER_GLOBAL_STEP += 1 
            if SEQUENCER_GLOBAL_STEP > 7:
                SEQUENCER_GLOBAL_STEP = 0
            #print(SEQUENCER_GLOBAL_STEP)
            
            delay= te - d

            #print(f"Beat {i+1}: Interval = {te}s, Error = {delay}s correction_factor = {wait_time-d}")
            calculated = False
            
        else:

            if not calculated:
                # Update PID controller
                correction = pid.update(delay, d) # d hier der richtige wert?????
                            
                # Apply correction to wait time
                wait_time = max(0, d - correction)
                calculated = True
            else:
                pass

if __name__ == "__main__":
    # Start the sound playing thread (metronome)
    sound_thread = threading.Thread(target=render)
    sound_thread.daemon = True
    sound_thread.start()

    # Start a thread to listen for keypresses
    listener_thread = threading.Thread(target=listen_for_keypress)
    listener_thread.daemon = True  # Daemon thread so it exits when the main program exits
    listener_thread.start()

    # Start the main loop
    main_loop()
