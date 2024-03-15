import os

import cv2
import matplotlib.pyplot as plt

# Get the absolute path to the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the relative path to the image file
image_path = os.path.join(script_dir, 'penguin.jpg')

# Read the image
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Task 2: Apply global thresholding to create a binary image (cv2.threshold).
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Task 3: Experiment with different threshold values and observe the effects.
# _, binary_image_low = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
# _, binary_image_high = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

# EX2
# Task 1: Apply erosion to the binary image created in Exercise 1 (cv2.erode).
kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

eroded_square = cv2.erode(binary_image, kernel_square, iterations=1)
eroded_circle = cv2.erode(binary_image, kernel_circle, iterations=1)

# Task 2: Apply dilation to the same image (cv2.dilate).
dilated_square = cv2.dilate(binary_image, kernel_square, iterations=1)
dilated_circle = cv2.dilate(binary_image, kernel_circle, iterations=1)

# Task 3: Perform opening and closing operations (cv2.morphologyEx with cv2.MORPH_OPEN and cv2.MORPH_CLOSE).
opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_square)
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_square)

# Display original and processed images
plt.subplot(3, 3, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(3, 3, 2), plt.imshow(gray_image, cmap='gray'), plt.title('Grayscale Image')
plt.subplot(3, 3, 3), plt.imshow(binary_image, cmap='gray'), plt.title('Binary Image')
plt.subplot(3, 3, 4), plt.imshow(eroded_square, cmap='gray'), plt.title('Eroded (Square)')
plt.subplot(3, 3, 5), plt.imshow(eroded_circle, cmap='gray'), plt.title('Eroded (Circle)')
plt.subplot(3, 3, 6), plt.imshow(dilated_square, cmap='gray'), plt.title('Dilated (Square)')
plt.subplot(3, 3, 7), plt.imshow(dilated_circle, cmap='gray'), plt.title('Dilated (Circle)')
plt.subplot(3, 3, 8), plt.imshow(opened_image, cmap='gray'), plt.title('Opened Image')
plt.subplot(3, 3, 9), plt.imshow(closed_image, cmap='gray'), plt.title('Closed Image')
plt.gcf().set_dpi(300)
plt.show()
