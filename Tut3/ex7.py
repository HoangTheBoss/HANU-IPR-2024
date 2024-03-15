import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Get the absolute path to the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the relative path to the image file
image_path = os.path.join(script_dir, 'test.jpg')

# Read the image
image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)

red_only = image.copy()
red_only[:, :, 1] = 0
red_only[:, :, 2] = 0

green_only = image.copy()
green_only[:, :, 0] = 0
green_only[:, :, 2] = 0

blue_only = image.copy()
blue_only[:, :, 0] = 0
blue_only[:, :, 1] = 0

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

hue_channel = hsv_image[:, :, 0]
saturation_channel = hsv_image[:, :, 1]
value_channel = hsv_image[:, :, 2]

flat_value_channel = 255 * np.ones_like(hue_channel)

modified_hue_saturation = cv2.merge([hue_channel, saturation_channel, flat_value_channel])
hue_saturation_rgb = cv2.cvtColor(modified_hue_saturation, cv2.COLOR_HSV2RGB)

ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

y_channel = ycbcr_image[:, :, 0]
cb_channel = ycbcr_image[:, :, 1]
cr_channel = ycbcr_image[:, :, 2]

cb_tinted = image.copy()
cb_tinted[:, :, 2] = (cb_channel - 128) * 2
cb_tinted[:, :, 0] = cb_tinted[:, :, 1] = (128 - cb_channel) * 2

cr_tinted = image.copy()
cr_tinted[:, :, 0] = (cr_channel - 128) * 2
cr_tinted[:, :, 2] = cr_tinted[:, :, 1] = (128 - cr_channel) * 2

plt.subplot(4, 3, 2), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')

plt.subplot(4, 3, 4), plt.imshow(red_only), plt.title('Red Only')
plt.subplot(4, 3, 5), plt.imshow(green_only), plt.title('Green Only')
plt.subplot(4, 3, 6), plt.imshow(blue_only), plt.title('Blue Only')

plt.subplot(4, 3, 7), plt.imshow(hue_channel, cmap='hsv'), plt.title('Hue')
plt.subplot(4, 3, 8), plt.imshow(hue_saturation_rgb), plt.title('Hue+Sat')
plt.subplot(4, 3, 9), plt.imshow(value_channel, cmap='gray'), plt.title('Val')

plt.subplot(4, 3, 10), plt.imshow(y_channel, cmap='gray'), plt.title('Y')
plt.subplot(4, 3, 11), plt.imshow(cb_tinted), plt.title('Cb')
plt.subplot(4, 3, 12), plt.imshow(cr_tinted), plt.title('Cr')

plt.gcf().set_dpi(300)
plt.show()