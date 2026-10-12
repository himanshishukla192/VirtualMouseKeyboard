import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# ---------------- pyautogui setup ----------------
pyautogui.PAUSE = 0         
pyautogui.FAILSAFE = True    

# ---------------- tunable settings ----------------
CAM_W, CAM_H = 640, 480
FRAME_MARGIN = 100           # hand only needs to cover the inner box; edges of screen become easy to reach

MIN_ALPHA, MAX_ALPHA = 0.15, 0.70   # smoothing: slow hand = heavy smoothing, fast hand = responsive
FAST_SPEED_PX = 300                 # screen px/frame at which smoothing becomes fully responsive

PINCH_ON = 0.22              # pinch distance / palm size  -> click
PINCH_OFF = 0.32             
CLICK_COOLDOWN = 0.30        # seconds
LOCK_HISTORY = 3             
                             # keep small: too large and fast moves click short of small targets like the X button
HOVER_DELAY = 0.06          
CLICK_HOLD = 0.06           
SCROLL_HOLD_FRAMES = 3      
SCROLL_DEADZONE = 0.03       # ignore tiny hand movement (fraction of frame height)
SCROLL_SPEED = 600           # bigger = faster scroll
SCROLL_SMOOTH = 0.4


def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def reliable_click(x, y):
    """Move, wait for hover, press, hold briefly, release.
    pyautogui.click() presses and releases within microseconds, which window
    buttons (close/minimize/maximize) often ignore."""
    pyautogui.moveTo(x, y)
    time.sleep(HOVER_DELAY)
    pyautogui.mouseDown()
    time.sleep(CLICK_HOLD)
    pyautogui.mouseUp()


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path="hand_landmarker.task",
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    screen_w, screen_h = pyautogui.size()

    smooth_x = smooth_y = None
    history = deque(maxlen=LOCK_HISTORY)
    pinching = False
    locked_pos = None
    last_click = 0.0

    scrolling = False
    scroll_frames = 0
    scroll_anchor = None
    scroll_offset = 0.0

    last_ts = 0
    status = "No hand"

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # timestamps must be strictly increasing
        ts = max(int(time.monotonic() * 1000), last_ts + 1)
        last_ts = ts
        result = detector.detect_for_video(mp_image, ts)

        # draw the active area
        cv2.rectangle(frame, (FRAME_MARGIN, FRAME_MARGIN),
                      (CAM_W - FRAME_MARGIN, CAM_H - FRAME_MARGIN), (255, 0, 255), 2)

        if not result.hand_landmarks:
            # hand lost: reset everything so nothing jumps when it comes back
            smooth_x = smooth_y = None
            history.clear()
            pinching = False
            scrolling = False
            scroll_frames = 0
            scroll_anchor = None
            status = "No hand"
        else:
            hand = result.hand_landmarks[0]
            index_tip, thumb_tip = hand[8], hand[4]

            # ---- gesture detection ----
            index_up = hand[8].y < hand[6].y
            middle_up = hand[12].y < hand[10].y
            ring_down = hand[16].y > hand[14].y
            pinky_down = hand[20].y > hand[18].y
            scroll_gesture = index_up and middle_up and ring_down and pinky_down

            palm = dist(hand[0], hand[9]) or 1e-6      # makes pinch independent of hand distance
            pinch_ratio = dist(index_tip, thumb_tip) / palm

            # ---- SCROLL ----
            if scroll_gesture:
                scroll_frames += 1
                if scroll_frames >= SCROLL_HOLD_FRAMES:
                    y = hand[9].y
                    if not scrolling:
                        scrolling = True
                        scroll_anchor = y
                        scroll_offset = 0.0
                    
                    raw = scroll_anchor - y            
                    scroll_offset += (raw - scroll_offset) * SCROLL_SMOOTH
                    if abs(scroll_offset) > SCROLL_DEADZONE:
                        sign = 1 if scroll_offset > 0 else -1
                        amount = int((abs(scroll_offset) - SCROLL_DEADZONE) * SCROLL_SPEED) * sign
                        if amount:
                            pyautogui.scroll(amount)
                    status = "SCROLL"
            else:
                if scrolling:
                    smooth_x = smooth_y = None          
                scrolling = False
                scroll_frames = 0
                scroll_anchor = None

            # ---- CURSOR + CLICK (not while scrolling) ----
            if not scrolling and not scroll_gesture:
                # map the inner camera box to the whole screen
                px = np.interp(index_tip.x * CAM_W, (FRAME_MARGIN, CAM_W - FRAME_MARGIN), (0, screen_w))
                py = np.interp(index_tip.y * CAM_H, (FRAME_MARGIN, CAM_H - FRAME_MARGIN), (0, screen_h))
                px = float(np.clip(px, 2, screen_w - 3))
                py = float(np.clip(py, 2, screen_h - 3))

                if smooth_x is None:
                    smooth_x, smooth_y = px, py
                else:
                    speed = ((px - smooth_x) ** 2 + (py - smooth_y) ** 2) ** 0.5
                    alpha = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * min(speed / FAST_SPEED_PX, 1.0)
                    smooth_x += (px - smooth_x) * alpha
                    smooth_y += (py - smooth_y) * alpha

                if not pinching:
                    history.append((smooth_x, smooth_y))

                now = time.monotonic()
                if pinch_ratio < PINCH_ON and not pinching and now - last_click > CLICK_COOLDOWN:
                    pinching = True
                    last_click = now
                    # click where the cursor was BEFORE the fingers moved together
                    locked_pos = history[0] if history else (smooth_x, smooth_y)
                    reliable_click(int(locked_pos[0]), int(locked_pos[1]))
                    status = "CLICK"
                elif pinch_ratio > PINCH_OFF:
                    if pinching:
                        smooth_x, smooth_y = locked_pos   # resume smoothly from the click spot
                        history.clear()
                    pinching = False

                if pinching:
                    pyautogui.moveTo(int(locked_pos[0]), int(locked_pos[1]))
                else:
                    pyautogui.moveTo(int(smooth_x), int(smooth_y))
                    status = "MOVE"

            cv2.circle(frame, (int(index_tip.x * CAM_W), int(index_tip.y * CAM_H)), 8, (0, 255, 0), -1)

        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("Virtual Mouse (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()