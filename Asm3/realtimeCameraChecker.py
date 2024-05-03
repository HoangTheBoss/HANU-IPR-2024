import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

# Initialize main window
root = tk.Tk()
root.title("Object Detection")

# Frame for displaying the video
frame_display = ttk.Label(root)
frame_display.grid(row=0, column=0, padx=10, pady=10)

# Status label for displaying information
status_label = ttk.Label(root, text="Press 'Start' to begin detection", wraplength=400)
status_label.grid(row=2, column=0, padx=10, pady=10)


# Load YOLO model and classes
cfg_path = "yolov3.cfg"
weights_path = "yolov3.weights"
classes_path = "coco.names"
with open(classes_path, "r") as f:
    classes = f.read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
input_size = (128, 128)

# Global variables
cap = None
detection_active = False


# Function to process frames and overlay detections
def process_and_overlay(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, input_size, swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    unconnected_layers = net.getUnconnectedOutLayers()
    output_layers = [layer_names[idx - 1] for idx in unconnected_layers]
    detections = net.forward(output_layers)

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Calculate color based on confidence: red (low confidence) to green (high confidence)
                if confidence < 0.5:
                    # Below 50% confidence, always red
                    color = (0, 0, 255)
                else:
                    # Scale from 50% to 100% confidence: red to green
                    green = int((confidence - 0.5) * 2 * 255)
                    red = 255 - green
                    color = (0, green, red)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


# Function to update UI with processed frames
def update_ui():
    global cap, detection_active
    if detection_active and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = process_and_overlay(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image = Image.fromarray(frame)
            image_tk = ImageTk.PhotoImage(image=image)
            frame_display.imgtk = image_tk
            frame_display.configure(image=image_tk)
        frame_display.after(10, update_ui)

# Function to toggle video capture and processing
def toggle_detection():
    global detection_active, cap
    if detection_active:
        detection_active = False
        start_button.config(text="Start")
        status_label.config(text="Detection stopped. Press 'Start' to resume.")
        if cap.isOpened():
            cap.release()
    else:
        detection_active = True
        start_button.config(text="Stop")
        status_label.config(text="Detection running...")
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        update_ui()

# Button to start/stop detection
start_button = ttk.Button(root, text="Start", command=toggle_detection)
start_button.grid(row=1, column=0, padx=10, pady=10)

# Ensure the program ends cleanly
def on_closing():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the Tkinter event loop
root.mainloop()
