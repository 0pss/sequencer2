import os
import platform
import time

import subprocess
from typing import Tuple

import threading
import time
from time import perf_counter, sleep
import numpy as np
import simpleaudio as sa
import struct
from typing import List, Tuple

from utils import *
from globals import *
from I2CController import I2CController, dummy_I2CController
from PIDController import PIDController
from render import *

from multiprocessing import Process, Value, Array, Manager
import os 
from stat_server import serve_with_realtime_priority
 
from managers import SequencerState


def main_loop(i2c: I2CController, sound_process, stats_process, state: SequencerState):
    # Load samples
    raw_samples = load_n_samples("./", 4)
    state.raw_samples.extend(raw_samples)
    
    # Initialize timing variables
    delay = d = wait_time = 60 / state.bpm.value
    print(f'{60 / delay} bpm')
    
    a = perf_counter()
    calculated = True
    pid = PIDController(Kp=0.5, Ki=0.1, Kd=0.01)
    sound_process.start()
    stats_process.start()

    
    while not state.stopped.value:
        b = perf_counter()
        te = abs(b-a)
        
        if (te > wait_time) or a == 0:

            #print(te)

            a = perf_counter()
            #print("tick ", state.sequencer_global_step.value)
            
            # Play audio
            state.sequencer_audio[state.sequencer_global_step.value].play()
            
            # Update step with atomic operation
            with state.sequencer_global_step.get_lock():
                state.sequencer_global_step.value = (state.sequencer_global_step.value + 1) % 16
            
            delay = te - d
            calculated = False
        else:
            if not calculated:
                # Send current position to Arduino
                i2c.send_position(state.sequencer_global_step.value)
                i2c.get_bpm()
                new_bpm = i2c.current_bpm
                
                correction = pid.update(delay, d)
                wait_time = max(0, d - correction)
                calculated = True
                
                # Update BPM if changed
                if new_bpm != state.bpm.value:
                    with state.bpm.get_lock():
                        state.bpm.value = new_bpm
                        d = 60/np.max([new_bpm, 1])
                    print(f'New BPM: {new_bpm}')





def main():
    print(f"Running commit: {get_git_commit_hash()}")
    available_cores = os.cpu_count()
    print(f"Available cores: {available_cores}")

    #import simpleaudio.functionchecks as fc

    #fc.run_all()

    
    # Initialize shared state
    state = SequencerState()
    
    # Create appropriate controller based on platform
    if not is_raspberry_pi():
        print("Not running on a Raspberry Pi.")
        sound_process = Process(target=render_with_realtime_priority, args=(state,), daemon=True)
        stats_process = Process(target=serve_with_realtime_priority, args=(state,), daemon=True)
        i2c = dummy_I2CController()
    else:
        print("Running on a Raspberry Pi.")
        import board
        import busio
        import smbus2 as smbus
        from smbus2 import SMBus, i2c_msg
        i2c = I2CController()
        stats_process = Process(target=serve_with_realtime_priority, args=(state,), daemon=True)

        sound_process = Process(target=render_with_realtime_priority, args=(state,), daemon=True)
    
    # Start the main loop
    main_loop(i2c, sound_process, stats_process, state)

if __name__ == "__main__":
    main()