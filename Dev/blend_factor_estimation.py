import cv2
import numpy as np


def get_gradients(image, ksize):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    return np.array([gx, gy])


# This is to estimate "C" the image blend factor to help initialize alpha
def blend_factor_estimation(images, poisson_est, matte):
    # Adapted from:
    # https://github.com/rohitrango/automatic-watermark-detection/blob/master/src/watermark_reconstruct.py
    # (free to use for academic/research purposes)
    # Changed to work for grayscale images

    num_images = images.shape[0]
    Jm = (images - poisson_est)
    gx_jm = np.zeros(images.shape)
    gy_jm = np.zeros(images.shape)

    for i in range(num_images):
        gx_jm[i], gy_jm[i] = get_gradients(Jm[i], ksize=1)

    Jm_grad = np.sqrt(np.add(np.square(gx_jm), np.square(gy_jm)))
    est_Ik = matte * np.median(images, axis=0)
    gx_estIk, gy_estIk = get_gradients(est_Ik, ksize=3)
    estIk_grad = np.sqrt(np.add(np.square(gx_estIk), np.square(gy_estIk)))

    C = np.sum(Jm_grad[:, :, :] * estIk_grad[:, :]) / np.sum(np.square(estIk_grad[:, :])) / num_images

    return C, est_Ik
