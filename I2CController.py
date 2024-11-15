import globals
from utils import *
from multiprocessing import Process, Lock, Array, Value
from typing import Tuple
from smbus2 import SMBus, i2c_msg
from time import sleep
import ctypes

class I2CController(Process):

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
        """Initialize the I2C controller process."""
        Process.__init__(self, daemon=True)
        
        # I2C setup
        self.bus_number = bus_number
        self.arduino_address = arduino_address
        self.mpr121_address1 = 0x5A
        self.mpr121_address2 = 0x5B
        
        # Shared state setup using multiprocessing primitives
        self.current_bpm = Value(ctypes.c_int, 120)
        # Create boolean arrays for touch sensors (using c_bool for atomic operations)
        self.touch_data1 = Array(ctypes.c_bool, [False] * 12)
        self.touch_data2 = Array(ctypes.c_bool, [False] * 12)
        self.running = Value(ctypes.c_bool, True)
        self._lock = Lock()
        
        # The SMBus instance will be initialized in run() since it needs to be in the child process
        self.bus = SMBus(bus_number)

    def _init_mpr121(self, address):
        """Initialize a single MPR121 sensor with default configuration."""
        try:
            with self._lock:
                # Soft reset
                self.bus.write_byte_data(address, self.ELE_CFG_REG, 0x00)
                
                # Configure touch and release thresholds
                for i in range(12):
                    self.bus.write_byte_data(address, self.ELE0_T + i*2, 12)
                    self.bus.write_byte_data(address, self.ELE0_R + i*2, 6)
                
                # Configure baseline filtering
                self.bus.write_byte_data(address, self.MHD_R, 0x01)
                self.bus.write_byte_data(address, self.NHD_R, 0x01)
                self.bus.write_byte_data(address, self.NCL_R, 0x00)
                self.bus.write_byte_data(address, self.FDL_R, 0x00)
                
                # Enable electrodes and auto configuration
                self.bus.write_byte_data(address, self.ELE_CFG_REG, 0x0C)
                self.bus.write_byte_data(address, self.AUTO_CONFIG_0, 0x0B)
                self.bus.write_byte_data(address, self.AUTO_CONFIG_1, 0x9F)
                
        except IOError as e:
            print(f"Error initializing MPR121 at address 0x{address:02X}: {e}")
            raise

    def run(self):
        """Main process loop - continuously updates sensor data and BPM."""
                
        # Initialize MPR121 sensors
        self._init_mpr121(self.mpr121_address1)
        self._init_mpr121(self.mpr121_address2)
        
        while self.running.value:
            try:
                print("reading sensors?")
                # Update touch sensors
                with self._lock:
                    status1 = self.bus.read_word_data(self.mpr121_address1, self.TOUCH_STATUS_REG)
                    status2 = self.bus.read_word_data(self.mpr121_address2, self.TOUCH_STATUS_REG)
                
                # Update shared arrays
                for i in range(12):
                    self.touch_data1[i] = bool(status1 & (1 << i))
                    self.touch_data2[i] = bool(status2 & (1 << i))
                
                # Update BPM
                with self._lock:
                    msg = i2c_msg.read(self.arduino_address, 2)
                    self.bus.i2c_rdwr(msg)
                    
                    mes = [msg.buf[k] for k in range(msg.len)]
                    mes_bytes = b''.join(mes)
                    result = int.from_bytes(mes_bytes, byteorder='little')
                    
                    if result > 2**14:
                        result -= 2**16
                    
                    self.current_bpm.value = 120 + result

                    print("yes, all read")
                
            except Exception as e:
                print(f"Error in I2C process: {e}")
            
            sleep(0.1)  # Small delay to prevent busy-waiting
        
        # Clean up
        self.bus.close()

    def read_touch_sensors(self) -> Tuple[list, list]:
        """Non-blocking read of touch sensor status."""
        # Convert shared arrays to regular lists for return
        return (list(self.touch_data1), list(self.touch_data2))

    def get_bpm(self) -> int:
        """Non-blocking read of current BPM."""
        return self.current_bpm.value

    def send_position(self, position: int):
        """Send position update to Arduino for LED display."""
        try:
            with self._lock:
                data = position & 0x3F
                print(f"sending position: {position} and data: {data}")
                self.bus.write_i2c_block_data(self.arduino_address, 0x01, [data])
        except Exception as e:
            print(f"I2C write error (position): {e}")

    def stop(self):
        """Stop the I2C process and clean up."""
        self.running.value = False
        self.join()

    def __del__(self):
        """Ensure proper cleanup on object destruction."""
        if hasattr(self, 'running') and self.running.value:
            self.stop()


class dummy_I2CController(Process):

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
        """Initialize the I2C controller process."""
        Process.__init__(self, daemon=True)
        
        # I2C setup
        self.bus_number = bus_number
        self.arduino_address = arduino_address
        self.mpr121_address1 = 0x5A
        self.mpr121_address2 = 0x5B
        
        # Shared state setup using multiprocessing primitives
        self.current_bpm = 144
        # Create boolean arrays for touch sensors (using c_bool for atomic operations)
        self.touch_data1 = Array(ctypes.c_bool, [False] * 12)
        self.touch_data2 = Array(ctypes.c_bool, [False] * 12)
        self.running = Value(ctypes.c_bool, True)
        self._lock = Lock()
        
        # The SMBus instance will be initialized in run() since it needs to be in the child process
        self.bus = None

    def run(self):
        """Main process loop - continuously updates sensor data and BPM."""
        
        while self.running.value:
            pass
            
            sleep(0.1)  # Small delay to prevent busy-waiting
        
        # Clean up
        self.bus.close()

    def read_touch_sensors(self) -> Tuple[list, list]:
        """Non-blocking read of touch sensor status."""
        # Convert shared arrays to regular lists for return
        return (list(self.touch_data1), list(self.touch_data2))

    def get_bpm(self) -> int:
        """Non-blocking read of current BPM."""
        return self.current_bpm

    def send_position(self, position: int):
        """Send position update to Arduino for LED display."""
        pass

    def stop(self):
        """Stop the I2C process and clean up."""
        self.running.value = False
        self.join()

    def __del__(self):
        """Ensure proper cleanup on object destruction."""
        if hasattr(self, 'running') and self.running.value:
            self.stop()
