import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('penguin.jpg')

# Apply Gaussian blur with different kernel sizes
blurred_image_9x9 = cv2.GaussianBlur(image, (9, 9), 0)
blurred_image_15x15 = cv2.GaussianBlur(image, (15, 15), 0)
blurred_image_21x21 = cv2.GaussianBlur(image, (21, 21), 0)

# Display original and blurred images
plt.figure(figsize=(10, 4))

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original')

plt.subplot(1, 4, 2)
plt.imshow(cv2.cvtColor(blurred_image_9x9, cv2.COLOR_BGR2RGB))
plt.title('Blurred (9x9)')

plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(blurred_image_15x15, cv2.COLOR_BGR2RGB))
plt.title('Blurred (15x15)')

plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(blurred_image_21x21, cv2.COLOR_BGR2RGB))
plt.title('Blurred (21x21)')

plt.gcf().set_dpi(300)
plt.tight_layout()
plt.show()
