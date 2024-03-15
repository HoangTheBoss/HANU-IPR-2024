import os

import cv2
import matplotlib.pyplot as plt

# Get the absolute path to the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the relative path to the image file
image_path = os.path.join(script_dir, 'penguin.jpg')

# Read the image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply the Sobel operator to find edges
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

# Compute the magnitude of the gradient
gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Display the original image and the edges
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 2, 2), plt.imshow(gradient_magnitude, cmap='gray'), plt.title('Edge Detection')

# Show the plot
plt.gcf().set_dpi(300)
plt.show()
