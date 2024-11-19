import os

from time import perf_counter, sleep
import numpy as np
import simpleaudio as sa
from typing import List, Tuple

from utils import *
from globals import *
from I2CController import I2CController, dummy_I2CController, i2c_with_realtime_priority
from PIDController import PIDController
from render import *

from multiprocessing import Process, Value, Array, Manager
import os 
from stat_server import serve_with_realtime_priority
 
from managers import SequencerState


def main_loop(sound_process, i2com_process, stats_process, state: SequencerState):
    
    i2com_process.start()
    
    # Load samples
    raw_samples = load_n_samples("./", 4)
    state.raw_samples.extend(raw_samples)

    bpm = state.bpm.value
    
    # Initialize timing variables
    delay = d = wait_time = 60 / bpm
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
            print("playing sound")
            state.sequencer_audio[state.sequencer_global_step.value].play()
            
            # Update step with atomic operation
            with state.sequencer_global_step.get_lock():
                state.sequencer_global_step.value = (state.sequencer_global_step.value + 1) % 16
            
            delay = te - d
            calculated = False
        else:
            if not calculated:
                              
                correction = pid.update(delay, d)
                wait_time = max(0, d - correction)
                calculated = True
                
                # Update BPM if changed
                if bpm != state.bpm.value:
                    d = 60/np.max([state.bpm.value, 1])
                    bpm = state.bpm
                    print(f'New BPM: {state.bpm.value}')





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
        i2com_process = Process(target=i2c_with_realtime_priority, args=(state,), daemon=True)

    else:
        print("Running on a Raspberry Pi.")
        
        stats_process = Process(target=serve_with_realtime_priority, args=(state,), daemon=True)
        i2com_process = Process(target=i2c_with_realtime_priority, args=(state,), daemon=True)
        sound_process = Process(target=render_with_realtime_priority, args=(state,), daemon=True)
    
    # Start the main loop
    main_loop(sound_process, i2com_process, stats_process, state)

if __name__ == "__main__":
    main()