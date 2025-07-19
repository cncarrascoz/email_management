#!/bin/bash

# This script sets up a Python virtual environment and installs the required packages.
rm -rf venv

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt