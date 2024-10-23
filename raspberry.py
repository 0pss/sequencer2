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
    print(f"Current Git commit hash: {get_git_commit_hash()}")


if is_raspberry_pi():
    print("Running on a Raspberry Pi.")
else:
    print("Not running on a Raspberry Pi.")