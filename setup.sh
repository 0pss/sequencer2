#!/bin/bash

# Update and upgrade system
echo "Updating system..."
apt-get update -y
apt-get upgrade -y

# Install Python3 and pip3 if not installed
echo "Installing Python3 and pip3..."
apt-get install python3-pip python3-venv -y

# Check and configure /boot/config.txt for I2C if not already set
CONFIG_FILE="/boot/config.txt"
I2C_CONFIG="dtparam=i2c_arm=on"
echo "Checking /boot/config.txt for I2C configuration..."

if grep -q "^$I2C_CONFIG" "$CONFIG_FILE"; then
    echo "I2C is already enabled in /boot/config.txt. Skipping modification..."
else
    echo "I2C is not enabled in /boot/config.txt. Adding configuration..."
    echo "$I2C_CONFIG" | tee -a "$CONFIG_FILE" > /dev/null
    if [ $? -eq 0 ]; then
        echo "I2C configuration added successfully."
    else
        echo "Failed to add I2C configuration to /boot/config.txt."
        exit 1
    fi
fi

# Enable I2C interface
echo "Enabling I2C interface..."
raspi-config nonint do_i2c 0


# Check if the virtual environment already exists
if [ -d "sequencer" ]; then
    echo "Virtual environment 'sequencer' already exists. Skipping creation..."
else
    # Create a virtual environment named 'sequencer'
    echo "Creating virtual environment 'sequencer'..."
    python3 -m venv sequencer
fi

# Activate the virtual environment
if [ -f "sequencer/bin/activate" ]; then
    . sequencer/bin/activate
else
    echo "Error: Virtual environment 'sequencer' not found."
    exit 1
fi

# Upgrade pip inside the virtual environment
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Error: 'requirements.txt' not found. Skipping dependencies installation..."
fi

# Provide instructions to run the Python script
echo -e "\nSetup complete!"
