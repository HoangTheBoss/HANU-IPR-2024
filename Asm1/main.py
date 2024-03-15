import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize_image_keep_aspect(image, target_height=1080):
    # Get current dimensions of the image
    height, width = image.shape[:2]

    # Calculate the scaling factor, keeping the aspect ratio
    # Only scale down, if the image is larger than the target height
    if height > target_height:
        scaling_factor = target_height / height
        new_width = int(width * scaling_factor)
        new_height = target_height

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_image
    else:
        # Return original image if it does not need to be scaled down
        return image


def screen_blend(img1, img2, opacity):
    # Convert images to float and normalize (0 to 1)
    img1 = img1.astype(np.float32) / 255
    img2 = img2.astype(np.float32) / 255 * (opacity / 100)
    # Apply Screen blend mode formula
    result = 1 - (1 - img1) * (1 - img2)
    # Convert back to 8 bits
    return (result * 255).astype(np.uint8)


def apply_bloom(image):
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # Original Image
    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    # Extract brightness channel
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv_image[:, :, 2]

    # Threshold A
    _, threshold_A = cv2.threshold(brightness, 0.6 * 255, 255, cv2.THRESH_BINARY)
    threshold_A_blurred = cv2.GaussianBlur(threshold_A, (0, 0), 150).astype(np.float32) / 255
    axs[0, 1].imshow(threshold_A_blurred, cmap='gray')
    axs[0, 1].set_title('Threshold A with Blur')
    axs[0, 1].axis('off')

    # Apply color to Threshold A using alpha (blurred threshold as mask)
    orange_color = np.array([50, 150, 255], dtype=np.float32)
    threshold_A_colored = np.zeros_like(image, dtype=np.float32)
    for i in range(3):  # For each color channel
        threshold_A_colored[:, :, i] = orange_color[i] * threshold_A_blurred
    axs[1, 0].imshow(cv2.cvtColor(threshold_A_colored.astype(np.uint8), cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title('Colored Threshold A (Orange)')
    axs[1, 0].axis('off')

    # Threshold B
    _, threshold_B = cv2.threshold(brightness, 0.8 * 255, 255, cv2.THRESH_BINARY)
    threshold_B_blurred = cv2.GaussianBlur(threshold_B, (0, 0), 50).astype(np.float32) / 255
    axs[1, 1].imshow(threshold_B_blurred, cmap='gray')
    axs[1, 1].set_title('Threshold B with Blur')
    axs[1, 1].axis('off')

    # Apply color to Threshold B using alpha (blurred threshold as mask)
    red_color = np.array([40, 70, 255], dtype=np.float32)
    threshold_B_colored = np.zeros_like(image, dtype=np.float32)
    for i in range(3):  # For each color channel
        threshold_B_colored[:, :, i] = red_color[i] * threshold_B_blurred
    axs[2, 0].imshow(cv2.cvtColor(threshold_B_colored.astype(np.uint8), cv2.COLOR_BGR2RGB))
    axs[2, 0].set_title('Colored Threshold B (Red)')
    axs[2, 0].axis('off')

    # Convert original image to float32 and blend using Screen blending mode
    blended_A = screen_blend(image, threshold_A_colored.astype(np.uint8), 50)
    blended_B = screen_blend(blended_A, threshold_B_colored.astype(np.uint8), 80)

    # Convert back to uint8 and show final result
    result = np.clip(blended_B, 0, 255).astype(np.uint8)
    axs[2, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axs[2, 1].set_title('Final Bloom Effect')
    axs[2, 1].axis('off')

    plt.tight_layout()
    plt.show()

    return result


# Load image
image_path = "test.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Resize image while keeping aspect ratio
resized_image = resize_image_keep_aspect(image, 1080)

# Apply bloom effect
bloom_image = apply_bloom(image)
