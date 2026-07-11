#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Installing requirements..."
pip install -r requirements.txt

echo "Fixing OpenCV for headless environment..."
# MediaPipe installs opencv-contrib-python which requires GUI libraries (libGL)
# We uninstall it and ensure only headless versions are installed
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-python-headless opencv-contrib-python-headless
