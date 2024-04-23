# import queue # we dont use queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
from pygrabber.dshow_graph import FilterGraph


def camera_change_handler(event):
    global cap
    selected_camera = graph.get_input_devices().index(device_sel_box.get())
    # cap.set(cv2.CAP_PROP_SETTINGS, 0)
    cap.release()
    cap = cv2.VideoCapture(selected_camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_SETTINGS, 1)
    ret, frame = cap.read()


def detect_objects():
    global frame, detections

    detections_buffer = []
    buffer_size = 5
    # Load face detection model
    detection_prototxt = "detection.prototxt"
    detection_weights = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    detection_net = cv2.dnn.readNetFromCaffe(detection_prototxt, detection_weights)
    # Loop
    while True:
        start_time = time.time()

        if frame is not None:
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
            detection_net.setInput(blob)
            current_detections = detection_net.forward()

            # Update buffer with current detections
            detections_buffer.append(current_detections)
            if len(detections_buffer) > buffer_size:
                detections_buffer.pop(0)

            # Calculate average detections
            avg_detections = np.mean(np.array(detections_buffer), axis=0)
            detections = avg_detections

        elapsed_time = time.time() - start_time
        if elapsed_time < 1 / 15:
            time.sleep(1 / 15 - elapsed_time)


def calculate_cropped_frame(in_frame):
    global detections
    (h, w) = in_frame.shape[:2]
    min_confidence = 0.5

    # Initialize bounding box coordinates to values within image dimensions
    min_x, min_y, max_x, max_y = w, h, 0, 0

    # Iterate through all detections to find the bounding box that encompasses all faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Update the bounding box coordinates
            min_x = min(min_x, startX)
            min_y = min(min_y, startY)
            max_x = max(max_x, endX)
            max_y = max(max_y, endY)

    # Add padding to the bounding box
    padding_w = int((max_x - min_x) * 0.5)
    padding_h = int((max_y - min_y) * 0.5)
    min_x = max(0, min_x - padding_w)
    min_y = max(0, min_y - padding_h)
    max_x = min(w, max_x + padding_w)
    max_y = min(h, max_y + padding_h)

    # Calculate aspect ratio
    aspect_ratio = in_frame.shape[1] / in_frame.shape[0]

    # Calculate width and height of cropped frame with same aspect ratio
    width = max(int(max_x - min_x), int((max_y - min_y) * aspect_ratio))
    height = max(int((max_x - min_x) / aspect_ratio), int(max_y - min_y))

    # Ensure minimum dimensions
    width = max(width, 240 * aspect_ratio)  # Minimum width of 426 pixels at 16:9 (240p)
    height = max(height, 240)  # Minimum height of 240 pixels (240p)

    # Calculate center of the bounding box
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    # Calculate top-left corner of cropped frame
    top_left_x = max(0, int(center_x - width / 2))
    top_left_y = max(0, int(center_y - height / 2))

    # Calculate bottom-right corner of cropped frame
    bottom_right_x = min(in_frame.shape[1], int(center_x + width / 2))
    bottom_right_y = min(in_frame.shape[0], int(center_y + height / 2))

    # Crop frame
    cropped_frame = in_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    return cropped_frame



def runtime_loop():
    global ret, frame, sr
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if detections is not None:
            # Calculate cropped frame based on detection results, upscale
            cropped_frame = calculate_cropped_frame(frame.copy())
            upscaled_frame = cv2.resize(cropped_frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

            # Display cropped frame
            img = cv2.cvtColor(upscaled_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            panel.configure(image=img)
            panel.image = img


# load upscsale model
# sr = cv2.dnn_superres.DnnSuperResImpl_create()
# sr.readModel('FSRCNN-small_x3.pb')
# sr.setModel("fsrcnn", 3)

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_SETTINGS, 1)
ret, frame = cap.read()

# Var for passing detection results between threads
detections = None
cropped_frame = None

# Start detection thread
detection_thread = threading.Thread(target=detect_objects)
detection_thread.start()

runtime_thread = threading.Thread(target=runtime_loop)
runtime_thread.start()

# Start UI
root = tk.Tk()
root.title('Final Project: Auto Framing')
root.geometry('600x650')
# root.state('zoomed')
root.resizable(width=True, height=True)

graph = FilterGraph()
# print(graph.get_input_devices())#

# exit()

device_sel_box = ttk.Combobox(values=graph.get_input_devices())
device_sel_box.bind("<<ComboboxSelected>>", camera_change_handler)
device_sel_box.pack()

# start_btn = Button(root, text='Start!')
# start_btn.pack()

panel = tk.Label(root)
panel.pack(side="bottom", fill="both", expand="yes")

img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img)
img = ImageTk.PhotoImage(img)

panel.configure(image=img)
panel.image = img

root.mainloop()
cap.release()
