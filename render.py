from utils import *
from globals import *
import numpy as np
import simpleaudio as sa
from multiprocessing import Process, Value, Array, Manager
import os
import psutil 
import time
from raspberry import SequencerState


def render(state: SequencerState):
    """ Render the audio for columns that have changed. """

    print("RENDERING PROCESS STARTED")
    print(state.sequencer_changed)

    while True:
        for col_index, changed in enumerate(state.sequencer_changed):
            if changed > 0:  # Only process columns where a change occurred
                print("RENDERING")
                # Convert selected RAW_SAMPLES to numpy arrays
                active_samples_indices = [row_index for row_index, is_active in enumerate(state.sequencer_on) if is_active[col_index] == 1]
                selected_samples = [state.raw_samples[i] for i in active_samples_indices]
                
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
                state.sequencer_audio[col_index] = Numpy2Wave(mixed_samples)
                
                # Reset the change flag for this column
                state.sequencer_changed[col_index] = 0
        
        # Add a small delay to prevent high CPU usage
        time.sleep(0.1)  # Sleep for 10 milliseconds to reduce CPU load


def render_with_realtime_priority(state: SequencerState):
    process = psutil.Process(os.getpid())
    
    # Set CPU affinity to core 1
    process.cpu_affinity([1])
    
    # Optionally set high priority (Unix/Linux only)
    try:
        os.nice(-19)  # Highest priority
    except Exception as e:
        print(f"Could not set process priority: {e}")
        
    render(state)