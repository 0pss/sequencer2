#!/bin/bash

# Name of the sink (audio device) you want to use
AUDIO_DEVICE_NAME="your_usb_device_name"  # Replace this with your actual USB device name
# Path to your virtual environment
VENV_PATH="./sequencer"  # Replace this with the actual path to your virtual environment
# Path to your Python script
PYTHON_SCRIPT="./raspberry.py"  # Replace this with the actual path to your Python script

# 1. Set the desired audio output device using pactl
# Get the sink name from pactl list output
AUDIO_DEVICE_SINK=$(pactl list short sinks | grep "$AUDIO_DEVICE_NAME" | awk '{print $1}')

if [ -z "$AUDIO_DEVICE_SINK" ]; then
  echo "Error: Audio device '$AUDIO_DEVICE_NAME' not found."
  exit 1
fi

# Set the selected device as the default audio output
pactl set-default-sink "$AUDIO_DEVICE_SINK"
echo "Audio device set to: $AUDIO_DEVICE_NAME"

# 2. Activate the virtual environment
. "$VENV_PATH/bin/activate"

# 3. Start the Python script with the highest scheduling priority (-20)
# Requires root privileges for nice -20
echo "Starting Python script with highest priority..."
sudo nice -n -20 python "$PYTHON_SCRIPT"

# Deactivate the virtual environment after the script finishes
deactivate