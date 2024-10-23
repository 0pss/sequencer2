import os
import platform

def is_raspberry_pi():
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


import subprocess

def get_git_commit_hash():
    try:
        # Run git command to get the short commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
        return commit_hash
    except Exception as e:
        return f"Error retrieving Git commit hash: {e}"

if __name__ == "__main__":
    print(f"Running commit: {get_git_commit_hash()}")


if is_raspberry_pi():
    print("Running on a Raspberry Pi.")

    import time
    import board
    import busio
    import adafruit_mpr121

    # Initialize I2C bus and MPR121
    i2c = busio.I2C(board.SCL, board.SDA)
    mpr121 = adafruit_mpr121.MPR121(i2c)

    print("Touch sensor test. Press Ctrl-C to quit.")

    # Main loop to print touch status
    while True:
        for i in range(12):
            if mpr121[i].value:
                print(f"Input {i} is touched.")
        time.sleep(0.1)



else:
    print("Not running on a Raspberry Pi.")