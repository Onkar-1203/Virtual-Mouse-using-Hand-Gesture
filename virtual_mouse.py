#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as scb
import time
from comtypes import CLSCTX_ALL

# Initialize
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

prev_x, prev_y = screen_width // 2, screen_height // 2
last_click_time = 0
dragging = False
click_cooldown = 1.0  # seconds
double_click_threshold = 0.3

current_volume = 0.5
current_brightness = 50

vol_change_threshold = 0.01
bright_change_threshold = 2

is_volume_control_active = False
is_brightness_control_active = False

def set_volume(volume_level):
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume.iid, CLSCTX_ALL, None)
        volume = interface.QueryInterface(IAudioEndpointVolume)
        volume.SetMasterVolumeLevelScalar(volume_level, None)
    except Exception as e:
        print("Volume error:", e)

def set_brightness(level):
    try:
        scb.set_brightness(level)
    except Exception as e:
        print("Brightness error:", e)

def smooth_move(target_x, target_y, prev_x, prev_y, smooth_factor=0.2):
    return int(prev_x + (target_x - prev_x) * smooth_factor), int(prev_y + (target_y - prev_y) * smooth_factor)

def main_loop():
    global prev_x, prev_y, last_click_time, dragging
    global current_volume, current_brightness
    global is_volume_control_active, is_brightness_control_active

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        drawing_utils = mp.solutions.drawing_utils

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                frame = cv2.flip(frame, 1)
                frame_height, frame_width, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)

                if result.multi_hand_landmarks:
                    for hand in result.multi_hand_landmarks:
                        drawing_utils.draw_landmarks(frame, hand)
                        landmarks = hand.landmark
                        lm_list = [(int(lm.x * frame_width), int(lm.y * frame_height)) for lm in landmarks]

                        thumb_tip = np.array(lm_list[4])
                        index_tip = np.array(lm_list[8])
                        middle_tip = np.array(lm_list[12])
                        ring_tip = np.array(lm_list[16])
                        pinky_tip = np.array(lm_list[20])

                        target_x = int(screen_width / frame_width * index_tip[0])
                        target_y = int(screen_height / frame_height * index_tip[1])
                        prev_x, prev_y = smooth_move(target_x, target_y, prev_x, prev_y)
                        pyautogui.moveTo(prev_x, prev_y)

                        # Volume Control
                        dist_tp = np.linalg.norm(thumb_tip - pinky_tip)
                        if dist_tp < 50:
                            is_volume_control_active = True
                            is_brightness_control_active = False
                            new_volume = max(current_volume - 0.02, 0.0)
                            if abs(new_volume - current_volume) > vol_change_threshold:
                                current_volume = new_volume
                                set_volume(current_volume)
                        elif 50 <= dist_tp <= 100:
                            if is_volume_control_active:
                                new_volume = min(current_volume + 0.02, 1.0)
                                if abs(new_volume - current_volume) > vol_change_threshold:
                                    current_volume = new_volume
                                    set_volume(current_volume)

                        # Brightness Control
                        dist_tr = np.linalg.norm(thumb_tip - ring_tip)
                        if dist_tr < 30:
                            is_brightness_control_active = True
                            is_volume_control_active = False
                            new_brightness = max(current_brightness - 2, 0)
                            if abs(new_brightness - current_brightness) >= bright_change_threshold:
                                current_brightness = new_brightness
                                set_brightness(current_brightness)
                        elif 30 <= dist_tr <= 80:
                            if is_brightness_control_active:
                                new_brightness = min(current_brightness + 2, 100)
                                if abs(new_brightness - current_brightness) >= bright_change_threshold:
                                    current_brightness = new_brightness
                                    set_brightness(current_brightness)

                        # Scroll
                        dist_im = np.linalg.norm(index_tip - middle_tip)
                        if dist_im < 35:
                            if index_tip[1] < middle_tip[1]:
                                pyautogui.scroll(20)
                            elif index_tip[1] > middle_tip[1]:
                                pyautogui.scroll(-20)

                        # Drag and Drop
                        if np.linalg.norm(thumb_tip - index_tip) < 40:
                            if not dragging:
                                pyautogui.mouseDown()
                                dragging = True
                        else:
                            if dragging:
                                pyautogui.mouseUp()
                                dragging = False

                        # Single Click
                        if np.linalg.norm(thumb_tip - index_tip) < 30 and not dragging:
                            current_time = time.time()
                            if current_time - last_click_time > click_cooldown:
                                pyautogui.click()
                                last_click_time = current_time

                        # Double Click
                        if np.linalg.norm(thumb_tip - middle_tip) < 35:
                            current_time = time.time()
                            if current_time - last_click_time < double_click_threshold:
                                pyautogui.click(clicks=2)
                                last_click_time = 0
                            else:
                                last_click_time = current_time

                cv2.imshow("Virtual Mouse Control", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Guaranteed cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed.")

if __name__ == "__main__":
    main_loop()


# In[ ]:




