#!/bin/bash

# Update and upgrade system
echo "Updating system..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python3 and pip3 if not installed
echo "Installing Python3 and pip3..."
sudo apt-get install python3-pip python3-venv -y

# Enable I2C interface
echo "Enabling I2C interface..."
sudo raspi-config nonint do_i2c 0

# Create a virtual environment named 'sequencer'
echo "Creating virtual environment 'sequencer'..."
python3 -m venv sequencer

# Activate the virtual environment
source sequencer/bin/activate

# Upgrade pip inside the virtual environment
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies from requirements.txt
echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# Provide instructions to run the Python script
echo -e "\nSetup complete!"
