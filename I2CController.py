import time
import psutil
from managers import SequencerState
from utils import *
from multiprocessing import Process, Lock, Array, Value
from typing import Tuple
from smbus2 import SMBus, i2c_msg
from time import sleep
import ctypes

#  self.bus.close()

class InputEdgeDetector:
    def __init__(self, debounce_threshold=3):
        self.debounce_threshold = debounce_threshold
        self.touch_buffer = [[0] * 20 for _ in range(4)]  # Tracks stable states
        self.touch_counters = [[0] * 20 for _ in range(4)]  # Stability counters
        self.previous_state = [[0] * 20 for _ in range(4)]  # Tracks previous stable states

    def debounce_and_detect_edge(self, row, col, current_state):
        """
        Debounce logic combined with rising/falling edge detection.
        Returns:
            "rising" for a rising edge (0 -> 1),
            "falling" for a falling edge (1 -> 0),
            None if no edge is detected.
        """
        # Debounce logic
        if current_state:
            self.touch_counters[row][col] += 1
            if self.touch_counters[row][col] >= self.debounce_threshold:
                self.touch_buffer[row][col] = 1
        else:
            self.touch_counters[row][col] = 0
            self.touch_buffer[row][col] = 0

        # Edge detection
        edge_detected = None
        if self.touch_buffer[row][col] != self.previous_state[row][col]:
            if self.touch_buffer[row][col] == 1:
                edge_detected = "rising"
            else:
                edge_detected = "falling"

        # Update previous state
        self.previous_state[row][col] = self.touch_buffer[row][col]

        return edge_detected

def init(state: SequencerState):
    # MPR121 Register addresses
    MHD_R = 0x2B
    NHD_R = 0x2C
    NCL_R = 0x2D
    FDL_R = 0x2E
    MHD_F = 0x2F
    NHD_F = 0x30
    NCL_F = 0x31
    FDL_F = 0x32
    ELE0_T = 0x41
    ELE0_R = 0x42
    ELE1_T = 0x43
    ELE1_R = 0x44
    TTH = 0x41
    RTH = 0x42
    DEBOUNCE = 0x5B
    CONFIG1 = 0x5C
    CONFIG2 = 0x5D

    # Touch Status Register
    TOUCH_STATUS = 0x00

    bus_number: int = 1
        
    # I2C setup
    mpr121_addresses = [0x5A, 0x5B]
      

    # The SMBus instance will be initialized in run() since it needs to be in the child process
    bus = SMBus(bus_number)

    for address in mpr121_addresses:
        try:
            bus.write_byte_data(address, 0x80, 0x63)
            time.sleep(0.1)
            
            # Stop mode
            bus.write_byte_data(address, 0x5E, 0x00)
            
            # Configuration registers
            bus.write_byte_data(address, MHD_R, 0x01)
            bus.write_byte_data(address, NHD_R, 0x01)
            bus.write_byte_data(address, NCL_R, 0x00)
            bus.write_byte_data(address, FDL_R, 0x00)
            
            # Filter configuration
            bus.write_byte_data(address, MHD_F, 0x01)
            bus.write_byte_data(address, NHD_F, 0x01)
            bus.write_byte_data(address, NCL_F, 0xFF)
            bus.write_byte_data(address, FDL_F, 0x02)
            
            # Touch and release thresholds
            for i in range(12):
                bus.write_byte_data(address, TTH + i*2, 20)  # Touch threshold
                bus.write_byte_data(address, RTH + i*2, 15)  # Release threshold
            
            # Debounce configuration
            bus.write_byte_data(address, DEBOUNCE, 0x11)  # 2 samples for touch and release
            
            # Auto configuration
            bus.write_byte_data(address, CONFIG1, 0x10)  # Default FFI of 6
            bus.write_byte_data(address, CONFIG2, 0x20)  # 0.5uS encoding, 1ms period
            
            # Enable all electrodes
            bus.write_byte_data(address, 0x5E, 0x8F)  # Run mode, all 12 electrodes enabled
            
            
        except Exception as e:
            print(f"Error initializing MPR121 at address 0x{address:02X}: {str(e)}")

    print("DONE INIT MPR121")

    return bus

def send_position(bus, arduino_address, state: SequencerState):
    try:
        position = (state.sequencer_global_step.value +15) % 16 #TODO: Stupid fix for one-off error
        #print(f"sending position: {position} and data: {data}")
        
        data = position & 0x3F
        bus.write_i2c_block_data(arduino_address, 0x01, [data])
    except Exception as e:
        print(f"I2C write error (position): {e}")

def read_bpm(bus, arduino_address, state: SequencerState):
    try:
        msg = i2c_msg.read(arduino_address, 2)
        bus.i2c_rdwr(msg)
        
        mes = [msg.buf[k] for k in range(msg.len)]
        mes_bytes = b''.join(mes)
        result = int.from_bytes(mes_bytes, byteorder='little')
        
        if result > 2**14:
            result -= 2**16
        
       
        state.bpm.value = 120 + result 
            
    except Exception as e:
        print(f"Error in I2C (reading BPM): {e}")

def read_mprs(bus, state, edge_detector):
    mpr121_addresses = [0x5A, 0x5B]
    TOUCH_STATUS_REG = 0x00

    try:
        # Read touch statuses from both sensors
        status1 = bus.read_word_data(mpr121_addresses[0], TOUCH_STATUS_REG)
        status2 = bus.read_word_data(mpr121_addresses[1], TOUCH_STATUS_REG)


        # Map last 4 outputs of Sensor 2 to rows 1-4 in column 11
        for i in range(4):  # i corresponds to rows 1–4
            row_active = bool(status2 & (1 << (i + 8)))  # Check bits 8–11 of status2
            
            if row_active:  # Process only if row is active
                # Sensor 1: Map columns 0-11 for the active row
                for j in range(12):  # j corresponds to columns 0–11
                    touch_data1 = bool(status1 & (1 << j))  # Check bits 0–11 of status1
                    edge = edge_detector.debounce_and_detect_edge(i + 1, j, touch_data1)
                    if edge == "rising":
                        state.sequencer_on[i][j] ^= 1  # Toggle on rising edge
                        state.sequencer_changed[j] = 1

                # Sensor 2: Map columns 12–15 for the active row
                for j in range(4):  # j corresponds to columns 12–15
                    touch_data2 = bool(status2 & (1 << j))  # Check bits 0–3 of status2
                    edge = edge_detector.debounce_and_detect_edge(i + 1, j + 12, touch_data2)
                    if edge == "rising":
                        state.sequencer_on[i][j + 12] ^= 1  # Toggle on rising edge
                        state.sequencer_changed[j+12] = 1
                

    except Exception as e:
        print(f"Error in I2C (reading MPR): {e}")

def read_mprs_debug(bus, state, edge_detector):
    mpr121_addresses = [0x5A, 0x5B]
    TOUCH_STATUS_REG = 0x00
    bit_to_output = {i: 11 - i for i in range(12)}  # Reverse mapping for bits 0 to 11

    try:
        # Read touch statuses from both sensors
        status1 = bus.read_word_data(mpr121_addresses[1], TOUCH_STATUS_REG)
        status2 = bus.read_word_data(mpr121_addresses[0], TOUCH_STATUS_REG)
        print(status1, status2)
        for i in range(4):  # i corresponds to rows 1–4
            row_active = bool(status2 & (1 << (i)))  # Check bits 8–11 of status2
            if row_active:  # Process only if row is active
                # Sensor 1: Map columns 0-11 for the active row
                for j in range(12):  # j corresponds to columns 0–11
                    touch_data1 = bool(status1 & (1 << j))  # Check bits 0–11 of status1
                    edge = edge_detector.debounce_and_detect_edge(i + 1, j, touch_data1)
                    if edge == "rising":
                        col = bit_to_output[j]
                        state.sequencer_on[i][col] ^= 1  # Toggle on rising edge
                        state.sequencer_changed[col] = 1

                # Sensor 2: Map columns 12–15 for the active row
                for j in range(4):  # j corresponds to columns 12–15
                    touch_data2 = bool(status2 & (1 << j))  # Check bits 0–3 of status2
                    edge = edge_detector.debounce_and_detect_edge(i + 1, j + 12, touch_data2)
                    if edge == "rising":
                        col = bit_to_output[j]
                        state.sequencer_on[i][col + 12] ^= 1  # Toggle on rising edge
                        state.sequencer_changed[col+12] = 1

                for j in [5,6,7,8]:
                    touch_data2 = bool(status2 & (1 << j))  # Check bits 0–3 of status2
                    edge = edge_detector.debounce_and_detect_edge(i + 1, j + 12, touch_data2)
                    if edge == "rising":
                        col = bit_to_output[j]
                        
                    
                        print(j,col, "<<<<<< LIVE MODE BEBE")

    except Exception as e:
        print(f"Error in I2C (reading MPR): {e}")

def send_array(bus, arduino_address, state):
    """
    Write the 4x16 sequencer state to Arduino via I2C with timeout and retry logic.
    
    Args:
        bus: SMBus object
        arduino_address: I2C address of Arduino
        sequencer_on: List of 4 Arrays, each containing 16 integers (0 or 1)
        retries: Number of retry attempts
        timeout: Timeout in seconds for each attempt
    """
    try:
    
        # Start with command byte for sequencer states (0x02)
        data_bytes = bytearray([0x02])
        
        # Pack each row into 2 bytes (16 bits)
        for row_idx, row in enumerate(state.sequencer_on):
            byte1 = 0  # First 8 bits
            byte2 = 0  # Second 8 bits
            
            # Pack first 8 bits
            for i in range(8):
                if row[i]:
                    byte1 |= (1 << i)
            
            # Pack second 8 bits
            for i in range(8):
                if row[i + 8]:
                    byte2 |= (1 << i)
            
            data_bytes.append(byte1)
            data_bytes.append(byte2)
                    
        # Create and send write message
        msg = i2c_msg.write(arduino_address, data_bytes)
        bus.i2c_rdwr(msg)
        
        
    except Exception as e:
        print(f"Attempt failed: {e}")


def I2Ccommunicate(state: SequencerState):

    arduino_address: int = 0x08

    debouncer = InputEdgeDetector(debounce_threshold=1)

    bus = init(state)

    while True:
        # send position
        send_position(bus, arduino_address, state)
        sleep(0.01)
        # Read Touch
        read_mprs_debug(bus, state, debouncer)
        sleep(0.01)
        #read BPM
        read_bpm(bus, arduino_address, state)
        sleep(0.01)
        #send LED array
        send_array(bus, arduino_address, state)
        sleep(0.01)

def i2c_with_realtime_priority(state: SequencerState):
    process = psutil.Process(os.getpid())

    # Set CPU affinity to core 1
    process.cpu_affinity([3])

    # Optionally set high priority (Unix/Linux only)
    try:
        os.nice(-19)  # Highest priority
    except Exception as e:
        print(f"Could not set process priority: {e}")
        
    I2Ccommunicate(state)