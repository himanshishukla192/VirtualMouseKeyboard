Virtual Mouse Using Hand Gestures

A Python-based virtual mouse that uses real-time hand tracking through a webcam to control the computer cursor, click, and scroll without a physical mouse.

## Features

*Cursor Movement: Move the index finger to control the mouse pointer.
*Pinch-to-Click:  Bring the thumb and index finger together to click.
*Gesture-Based Scrolling: Raise the index and middle fingers while keeping the ring and pinky fingers down, then move the hand to scroll.

## Tech Stack

* Python
* OpenCV
* MediaPipe
* NumPy
* PyAutoGUI

## Requirements

* Python 3.12 recommended
* A working webcam
* macOS or another compatible desktop environment

## Setup and Usage

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Create and activate a virtual environment.

4. Install the required libraries:
   pip install -r requirements.txt
   
5. Make sure the `hand_landmarker.task` model file is available in the project directory.

6. Run the application:
   python3 -u main.py
 
7. Press `q` in the webcam window to exit.

## Project Goal

This project explores computer vision, hand landmark detection, gesture recognition, and desktop automation using Python.
