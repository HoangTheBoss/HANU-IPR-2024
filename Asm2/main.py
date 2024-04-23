import tkinter as tk
from tkinter import filedialog, Scale, Button
from PIL import Image, ImageTk
import numpy as np
import cv2


def openfilename():
    filename = filedialog.askopenfilename(title='Load Image for FFT Demo')
    return filename


def open_img():
    x = openfilename()
    global image
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
    img = ImageTk.PhotoImage(image)

    panel.configure(image=img)
    panel.image = img


def apply_filter(event=None):
    global image
    if image is not None:
        # Convert PIL Image to numpy array for processing
        image_np = np.array(image.convert('L'))
        _, _, filtered_image = fourier_filter(image_np, hp_scale.get())
        display_image = Image.fromarray(np.uint8(filtered_image)).convert('L')
        # display_image = display_image.resize((512, 512), Image.ANTIALIAS)
        img_display = ImageTk.PhotoImage(display_image)
        panel.configure(image=img_display)
        panel.image = img_display


def fourier_filter(image_array, cutoff_frequency):
    f_transform = np.fft.fft2(image_array)
    f_shift = np.fft.fftshift(f_transform)
    rows, cols = image_array.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 1
    f_shift_filtered = f_shift * mask
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return image_array, np.log(np.abs(f_shift)), img_back


root = tk.Tk()
root.title('Assignment 2: FFT Demo')
root.geometry('600x650')
# root.state('zoomed')
root.resizable(width=True, height=True)

image = None

load_btn = Button(root, text='Load Image', command=open_img)
load_btn.pack()

hp_scale = Scale(root, from_=1, to=500, resolution=1, orient=tk.HORIZONTAL, label="High Pass Filter Radius",
                 command=apply_filter, length=500)
hp_scale.pack()

panel = tk.Label(root)
panel.pack(side="bottom", fill="both", expand="yes")

root.mainloop()
