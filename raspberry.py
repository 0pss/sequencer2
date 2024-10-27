import os
import platform
import time
import board
import busio
import adafruit_mpr121
import subprocess
from typing import Tuple

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

class DualMPR121Handler:
    def __init__(self):
        self.i2c = busio.I2C(board.SCL, board.SDA)
        # Initialize two MPR121 sensors with different addresses
        # Default address is 0x5A, second sensor needs to be configured to 0x5B
        self.mpr121_1 = adafruit_mpr121.MPR121(self.i2c, address=0x5A)
        self.mpr121_2 = adafruit_mpr121.MPR121(self.i2c, address=0x5B)
        
    def read_sensors(self) -> Tuple[list, list]:
        """Read the status of all inputs from both sensors."""
        sensor1_status = [self.mpr121_1[i].value for i in range(12)]
        sensor2_status = [self.mpr121_2[i].value for i in range(12)]
        return sensor1_status, sensor2_status
    
    def print_touched_inputs(self):
        """Print which inputs are currently being touched on both sensors."""
        sensor1_status, sensor2_status = self.read_sensors()
        
        # Check sensor 1
        for i, touched in enumerate(sensor1_status):
            if touched:
                print(f"Sensor 1 - Input {i} is touched.")
                
        # Check sensor 2
        for i, touched in enumerate(sensor2_status):
            if touched:
                print(f"Sensor 2 - Input {i} is touched.")

def main():
    print(f"Running commit: {get_git_commit_hash()}")
    
    if not is_raspberry_pi():
        print("Not running on a Raspberry Pi.")
        return
    
    print("Running on a Raspberry Pi.")
    print("Dual touch sensor test. Press Ctrl-C to quit.")
    
    try:
        # Initialize the dual sensor handler
        sensor_handler = DualMPR121Handler()
        
        # Main loop
        while True:
            sensor_handler.print_touched_inputs()
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()