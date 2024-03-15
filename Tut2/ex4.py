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
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)


# Task 2: Apply histogram equalization to enhance the image contrast
equalized_image = cv2.equalizeHist(image)

gradient_magnitude = gradient_magnitude.astype(equalized_image.dtype)
_, binary_edges = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)

alpha = 0.5  # Adjust the blending strength
blended_image = cv2.add(equalized_image, binary_edges)
# blended_image = cv2.addWeighted(equalized_image, 1 - alpha, gradient_magnitude, alpha, 0)

# Display the original image, edges, and equalized image stacked vertically
plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(2, 2, 2), plt.imshow(binary_edges, cmap='gray'), plt.title('Edge Detection')
plt.subplot(2, 2, 3), plt.imshow(equalized_image, cmap='gray'), plt.title('Equalized Image')
plt.subplot(2, 2, 4), plt.imshow(blended_image, cmap='gray'), plt.title('Blended Image')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.gcf().set_dpi(300)
plt.show()

