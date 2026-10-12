Virtual Mouse Using Hand Gestures 🖐️

A Python-based virtual mouse that uses a webcam to track hand gestures and control the computer cursor, click, and scroll.

Demo 

[▶️ Watch the Virtual Mouse Demo](https://github.com/himanshishukla192/VirtualMouseKeyboard/blob/main/demo%20video....mov)



Features

- **Cursor Movement**: Move your index finger to control the mouse pointer.
- **Pinch-to-Click**: Bring your thumb and index finger together to click.
- **Gesture Based Scrolling**: Raise your index and middle fingers while keeping your ring and pinky fingers down, then move your hand to scroll.

Tech Stack

* Python
* OpenCV
* MediaPipe
* NumPy
* PyAutoGUI

Requirements

* Python 3.12 recommended
* A working webcam
* A compatible desktop environment
* The `hand_landmarker.task` model file

Setup and Usage

1. Clone or download this repository.

2. Open a terminal in the project directory.

3. Create and activate a virtual environment.

4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Download the compatible MediaPipe Hand Landmarker model and save it as `hand_landmarker.task` in the project directory. Follow the official [MediaPipe Hand Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).

6. Run the application:

   ```bash
   python3 -u main.py
   ```

7. Press `q` in the webcam window to exit.

Project Goal

This project explores computer vision, hand landmark detection, gesture recognition, and desktop automation using Python.
