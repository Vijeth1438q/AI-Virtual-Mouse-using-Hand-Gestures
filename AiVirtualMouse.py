import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import pyautogui
from collections import deque

# Webcam and Screen Setup
wCam, hCam = 640, 480
frameR = 100
smoothening = 7
motion_history = deque(maxlen=5)
scroll_y_history = deque(maxlen=7)
motion_threshold = 80

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
last_scroll_time = 0
scroll_delay = 0.15  # seconds between scrolls

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

zoom_mode = False
prev_zoom_len = None

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if lmList:
        x1, y1 = lmList[8][1:]   # Index
        x2, y2 = lmList[12][1:]  # Middle

        fingers = detector.fingersUp()

        # Gesture: Swipe Left/Right (4 fingers up, thumb down)
        if fingers == [0, 1, 1, 1, 1]:
            motion_history.append(x1)
            if len(motion_history) == motion_history.maxlen:
                dx = motion_history[-1] - motion_history[0]
                if dx > motion_threshold:
                    print("➡️ Swipe Right")
                    pyautogui.hotkey('ctrl', 'tab')
                    motion_history.clear()
                elif dx < -motion_threshold:
                    print("⬅️ Swipe Left")
                    pyautogui.hotkey('ctrl', 'shift', 'tab')
                    motion_history.clear()
        else:
            motion_history.clear()

        # Gesture: Move Cursor (Only index finger up)
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # Gesture: Click or Scroll (Index + Middle)
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            if length < 40:
                # Click
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
                print("🖱️ Click")
                time.sleep(0.2)
            elif length > 60 and fingers[3] == 0:
                # Scroll Up/Down using index finger Y movement
                y_pos = lmList[8][2]
                scroll_y_history.append(y_pos)
                if len(scroll_y_history) == scroll_y_history.maxlen:
                    dy = scroll_y_history[0] - scroll_y_history[-1]  # Y decreases = move up
                    current_time = time.time()
                    if abs(dy) > 15 and (current_time - last_scroll_time) > scroll_delay:
                        if dy > 0:
                            pyautogui.scroll(40)  # Scroll up
                            print("🔼 Scroll Up")
                        else:
                            pyautogui.scroll(-40)  # Scroll down
                            print("🔽 Scroll Down")
                        last_scroll_time = current_time

        # Gesture: Zoom (Pinch thumb and index)
        if fingers[0] == 1 and fingers[1] == 1:
            length, _, _ = detector.findDistance(4, 8, img, draw=False)
            if not zoom_mode:
                zoom_mode = True
                prev_zoom_len = length
            else:
                zoom_delta = length - prev_zoom_len
                if abs(zoom_delta) > 20:
                    if zoom_delta > 0:
                        pyautogui.hotkey('command', '+')
                        print("🔍 Zoom In")
                    else:
                        pyautogui.hotkey('command', '-')
                        print("🔎 Zoom Out")
                    prev_zoom_len = length
                    time.sleep(0.3)
        else:
            zoom_mode = False

    # Show FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    # Display window
    cv2.imshow("MacBook Touchpad - Virtual Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

