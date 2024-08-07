import cv2
import mss
import numpy as np
import pygetwindow as gw
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
import random
from ultralytics import YOLO
import torch

# Check for CUDA device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the custom YOLO model
model = YOLO("../computer_vision/runs/detect/train9/weights/best.pt").to(device)

# Dictionary to store colors for each class
class_colors = {}

def get_window_bbox(window_title="(12) TL vs C9 | FLY vs SR - 2024 LCS Summer Split - Week 6 Day 2 - YouTube - Opera"):
    """Get the bounding box of the specified window."""
    windows = gw.getWindowsWithTitle(window_title)
    if windows:
        target_window = windows[0]
        return (target_window.left, target_window.top, target_window.width, target_window.height)
    else:
        raise Exception("Specified window not found")

# Get the bounding box of the window
try:
    bbox = get_window_bbox()
except Exception as e:
    print(e)
    exit()

frame_queue = queue.Queue()
result_queue = queue.Queue()
display_event = threading.Event()

def capture_frames():
    sct = mss.mss()
    while True:
        sct_img = sct.grab({
            'left': bbox[0],
            'top': bbox[1],
            'width': bbox[2],
            'height': bbox[3]
        })
        frame = np.array(sct_img)
        frame_queue.put(frame)
        time.sleep(0.03)

def process_frame(frame):
    # Convert from BGRA to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    results = model(frame)

    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        conf = detection.conf[0]
        cls = detection.cls[0]

        class_name = model.names[int(cls)]
        if class_name not in class_colors:
            # Assign a new random color if the class is not already in the dictionary
            class_colors[class_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        label = f"{class_name} {conf:.2f}"
        color = class_colors[class_name]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def process_frames():
    while True:
        frame = frame_queue.get()
        result_frame = process_frame(frame)
        result_queue.put(result_frame)

def display_frames():
    display_event.wait()
    while True:
        frame = result_queue.get()
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.05)  # Delay display

# Start threads
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)
display_thread = threading.Thread(target=display_frames)

capture_thread.start()
process_thread.start()

# Wait for a few frames to be processed before starting display
time.sleep(10)
display_event.set()

display_thread.start()

capture_thread.join()
process_thread.join()
display_thread.join()

cv2.destroyAllWindows()
