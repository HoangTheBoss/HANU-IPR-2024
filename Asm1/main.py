import cv2
import numpy as np
import matplotlib.pyplot as plt


def soft_threshold(gray_image, threshold_ratio):
    threshold_value = threshold_ratio * 255
    mask = gray_image > threshold_value
    result_image = np.zeros_like(gray_image)
    result_image[mask] = gray_image[mask]
    return result_image


def resize_image_keep_aspect(image, target_height=1080):
    height, width = image.shape[:2]
    if height > target_height:
        scaling_factor = target_height / height
        return cv2.resize(image, (int(width * scaling_factor), target_height), interpolation=cv2.INTER_AREA)
    return image


def screen_blend(img1, img2, opacity):
    img1 = img1.astype(np.float32) / 255
    img2 = img2.astype(np.float32) / 255 * (opacity / 100)
    return ((1 - (1 - img1) * (1 - img2)) * 255).astype(np.uint8)


def apply_bloom(image):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv_image[:, :, 2]

    thresholds = [(0.6, [50, 150, 255], 100, 1), (0.8, [40, 70, 255], 30, 2)]
    for ratio, color, blur_size, row in thresholds:
        threshold = soft_threshold(brightness, ratio)
        axs[row, 0].imshow(threshold, cmap='gray')
        axs[row, 0].set_title(f'Threshold {chr(65 + row - 1)}')
        axs[row, 0].axis('off')

        threshold_blurred = cv2.GaussianBlur(threshold, (0, 0), blur_size).astype(np.float32) / 255
        axs[row, 1].imshow(threshold_blurred, cmap='gray')
        axs[row, 1].set_title(f'Threshold {chr(65 + row - 1)} with Blur')
        axs[row, 1].axis('off')

        threshold_colored = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            threshold_colored[:, :, i] = color[i] * threshold_blurred
        axs[row, 2].imshow(cv2.cvtColor(threshold_colored.astype(np.uint8), cv2.COLOR_BGR2RGB))
        axs[row, 2].set_title(f'Colored Threshold {chr(65 + row - 1)}')
        axs[row, 2].axis('off')

        if row == 1:
            blended = screen_blend(image, threshold_colored.astype(np.uint8), 50)
        else:
            blended = screen_blend(blended, threshold_colored.astype(np.uint8), 80)

    result = np.clip(blended, 0, 255).astype(np.uint8)
    axs[0, 2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axs[0, 2].set_title('Final Bloom Effect')
    axs[0, 2].axis('off')
    axs[0, 1].set_visible(False)
    plt.tight_layout()
    plt.show()

    return result


image_path = "test.jpg"
image = cv2.imread(image_path)
resized_image = resize_image_keep_aspect(image, 1080)
bloom_image = apply_bloom(resized_image)
