from common import *
import numpy as np
import cv2

BINARY_THRESH = 0.5
CANNY_THRESH = 0.5
MAX_UINT8_PIXEL = 255

"""
def _extract_bounding_box(watermark):
    white_pixels = np.where(watermark == MAX_UINT8_PIXEL)
    x1 = min(white_pixels[0])
    x2 = max(white_pixels[0])
    y1 = min(white_pixels[1])
    y2 = max(white_pixels[1])
    return (x1, y1, x2, y2) # Rectangle coordinates


def _debug_print_bounding_box(images, bounding_box):
    for i, img in enumerate(images):
        image = img.copy()
        image[bounding_box[0]:bounding_box[2], bounding_box[1]] = MAX_UINT8_PIXEL
        image[bounding_box[0]:bounding_box[2], bounding_box[3]] = MAX_UINT8_PIXEL
        image[bounding_box[0], bounding_box[1]:bounmatte, alphading_box[3]] = MAX_UINT8_PIXEL
        image[bounding_box[2], bounding_box[1]:bounding_box[3]] = MAX_UINT8_PIXEL
        save_image("boundingBox_{}.png".format(i), image)
"""


def get_gradients(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
    return gx, gy


def poisson_reconstruction(gradx, grady, kernel_size=3, num_iters=5000, h=0.75):
    # Adapted from:
    # https://github.com/rohitrango/automatic-watermark-detection/blob/master/src/estimate_watermark.py
    # (free to use for academic/research purposes)
    # Changed to work for grayscale images
    fxx = cv2.Sobel(gradx, cv2.CV_64F, 1, 0, ksize=kernel_size)
    fyy = cv2.Sobel(grady, cv2.CV_64F, 0, 1, ksize=kernel_size)

    laplacian = fxx + fyy
    m, n = laplacian.shape

    est = np.zeros(laplacian.shape)

    est[1:-1, 1:-1] = np.random.random((m - 2, n - 2))

    for i in range(num_iters):
        est[1:-1, 1:-1] = 0.25 * (
                    est[0:-2, 1:-1] + est[1:-1, 0:-2] + est[2:, 1:-1] + est[1:-1, 2:] - h * h * laplacian[1:-1, 1:-1])

    return est


def extract_watermark_outline(images, outline='binary'):
    # Get x and y gradients
    gradients_x = [0] * len(images)
    gradients_y = [0] * len(images)
    for i, img in enumerate(images):
        gradients_x[i], gradients_y[i] = get_gradients(img)

    # Get the absolute value of the gradients and combine
    wm_gradients_x = np.median(gradients_x, axis=0)
    wm_gradients_y = np.median(gradients_y, axis=0)
    wm_gradients_x_abs = np.absolute(wm_gradients_x)
    wm_gradients_y_abs = np.absolute(wm_gradients_y)
    watermark = np.sqrt(np.add(np.power(wm_gradients_x_abs, 2), np.power(wm_gradients_y_abs, 2)))

    # Normalize
    watermark = MAX_UINT8_PIXEL * (watermark - np.min(watermark)) / (np.max(watermark) - np.min(watermark))
    watermark = watermark.astype(np.uint8)

    # Outline Detection
    if outline == 'canny':
        watermark = cv2.Canny(watermark, MAX_UINT8_PIXEL * CANNY_THRESH, MAX_UINT8_PIXEL * CANNY_THRESH)
    if outline == 'binary':
        _, watermark = cv2.threshold(watermark, BINARY_THRESH, 1, cv2.THRESH_BINARY)
    else:
        raise (ValueError("Outline argument invalid"))

    return watermark, wm_gradients_x, wm_gradients_y


def initial_estimation(images):
    debug(print, "Starting initial estimation...")
    outline, gradx, grady = extract_watermark_outline(images)
    debug(print, "Starting poisson reconstruction...")
    W = poisson_reconstruction(gradx, grady)
    return W
