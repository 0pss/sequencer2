from utils import create_silent_wave


global SEQUENCER_AUDIO, SEQUENCER_CHANGED, RAW_SAMPLES, SEQUENCER_GLOBAL_STEP, BPM, STOPED, SEQUENCER_ON

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
