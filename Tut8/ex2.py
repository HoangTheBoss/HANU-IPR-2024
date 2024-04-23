import cv2
import numpy as np
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog, Scale, Button
from PIL import Image, ImageTk


def openfilename():
    filename = filedialog.askopenfilename(title='Load Image for FFT Demo')
    return filename


def open_img():
    x = openfilename()
    global image, hsv_image
    image = Image.open(x)

    # Calculate new dimensions while maintaining aspect ratio
    width, height = image.size
    if width < height:  # Short edge is width
        new_width = 512
        new_height = int((height / width) * new_width)
    else:  # Short edge is height
        new_height = 512
        new_width = int((width / height) * new_height)

    image = image.resize((new_width, new_height), Image.LANCZOS)

    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

    img = ImageTk.PhotoImage(image)
    panel.configure(image=img)
    panel.image = img


def filter_color(hue_map, select_hue, hue_range):
    # Calculate the absolute difference between the hue map and the selected hue
    hue_diff = np.abs(hue_map - select_hue)

    # Calculate the percentage of the hue difference relative to the full hue scale (0-179)
    # Adjust the calculation to use hue_range for controlling the aggressiveness
    hue_percentage = (hue_diff / hue_range) * 100
    hue_percentage = np.clip(hue_percentage, 0, 100)

    # Calculate the subtract map as a percentage of the original saturation
    # The subtraction is now based on how much percentage of the hue scale the difference represents
    # Adjusted by the hue_range to control the aggressiveness
    # subtract_map = (hue_percentage / hue_range) * 100
    # subtract_map = np.clip(subtract_map, 0, 100)  # Ensure percentages are within 0 to 100

    return hue_percentage.astype(np.uint8)


def apply_filter(event=None):
    global image, hsv_image
    if image is not None:
        # Create a copy of the hsv_image to keep the original hsv_image intact
        hsv_image_copy = hsv_image.copy()

        hue_map = hsv_image_copy[:, :, 0]
        saturation_subtract_map = filter_color(hue_map, hue_scale.get(), hue_range.get())

        saturation_channel_float = hsv_image_copy[:, :, 1].astype(np.float32)

        new_saturation_values = saturation_channel_float - (saturation_channel_float * saturation_subtract_map / 100)

        new_saturation_values_clipped = np.clip(new_saturation_values, 0, 255)

        hsv_image_copy[:, :, 1] = new_saturation_values_clipped.astype(np.uint8)

        result_image = cv2.cvtColor(hsv_image_copy, cv2.COLOR_HSV2BGR)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        display_image = Image.fromarray(result_image)

        img_display = ImageTk.PhotoImage(display_image)
        panel.configure(image=img_display)
        panel.image = img_display


# image_path = "test.jpg"
# image = cv2.imread(image_path)
# resized_image = resize_image_keep_aspect(image, 1080)
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# hue_map = hsv_image[:, :, 0]
#
# # Example usage
# select_hue = 90  # Example selected hue
# hue_range = 60  # Example hue range
# saturation_subtract_map = filter_color(hue_map, select_hue, hue_range)
#
# # Subtract the saturation subtract map from the saturation channel
# hsv_image[:, :, 1] = cv2.subtract(hsv_image[:, :, 1], saturation_subtract_map)
#
# # Convert back to BGR and save or display the image
# result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
# cv2.imwrite("filtered_image.jpg", result_image)

root = tk.Tk()
root.title('Tut 8 Exercise 2: Color Filter')
root.geometry('600x650')
# root.state('zoomed')
root.resizable(width=True, height=True)

image = None
hsv_image = None

load_btn = Button(root, text='Load Image', command=open_img)
load_btn.pack()

hue_scale = Scale(root, from_=0, to=179, resolution=1, orient=tk.HORIZONTAL, label="Hue Selection",
                 command=apply_filter, length=500)
hue_scale.pack()

hue_range = Scale(root, from_=1, to=180, resolution=1, orient=tk.HORIZONTAL, label="Hue Selection",
                 command=apply_filter, length=500)
hue_range.pack()

panel = tk.Label(root)
panel.pack(side="bottom", fill="both", expand="yes")

root.mainloop()