#!/bin/bash
# Installation script for rebuilt QADBNDM

# Create a virtual environment
echo "Creating a virtual environment..."
python -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "Installing the package..."
pip install -e .

# Ask if the user wants to install D-Wave support
read -p "Do you want to install D-Wave quantum annealing support? (y/n): " dwave
if [[ $dwave == "y" || $dwave == "Y" ]]; then
    echo "Installing D-Wave support..."
    pip install -e ".[dwave]"
fi

# Ask if the user wants to install development tools
read -p "Do you want to install development tools? (y/n): " dev
if [[ $dev == "y" || $dev == "Y" ]]; then
    echo "Installing development tools..."
    pip install -e ".[dev]"
fi

echo "Installation complete!"
echo "You can activate the virtual environment with: source venv/bin/activate"
echo "Run the CIFAR-10 example with: python examples/cifar_example.py" 