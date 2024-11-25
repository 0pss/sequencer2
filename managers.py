from multiprocessing import Process, Value, Array, Manager

from utils import create_silent_wave

class SequencerState:
    def __init__(self):
        self.manager = Manager()
        # Convert global variables to shared variables
        self.sequencer_audio = self.manager.list()
        for _ in range(16):
            self.sequencer_audio.append(create_silent_wave())

        self.sequencer_changed = Array('i', [1 for _ in range(16)])
        self.raw_samples = self.manager.list([])  # Will be populated with samples
        self.sequencer_global_step = Value('i', 0)
        self.bpm = Value('i', 120)
        self.stopped = Value('b', False)  # boolean
        self.live_mode = Value('b', False)  # Default to False
        
        # Create 4x16 grid of sequencer states
        # Use a list of Arrays for sequencer_on without Manager
        self.sequencer_on = [
            Array('i', [1] * 16),  # First row, all on
            *[Array('i', [0] * 16) for _ in range(3)]  # Other rows, all off
        ]
        