import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm

import initial_estimation
import single_image_matting
import image_watermark_decomposition
import blend_factor_estimation
import matte_update

from common import *
from image_watermark_decomposition import derivative


def load_image(image_path, color=cv2.IMREAD_GRAYSCALE):
    img = cv2.imread(image_path, color)
    return img


def load_images(directory):
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory)]
    images = np.array([load_image(i) for i in image_paths])
    return images


def save_image(image_path, image):
    cv2.imwrite(image_path, image)


def show_image(image):
    # im_255 = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    plt.imshow(image, cmap="gray")
    plt.show()


def normalize(image):
    return 1 * (image - np.min(image)) / (np.max(image) - np.min(image))


def optimization(W, alpha, images, itterations):
    W_hat = np.array(W)
    W_k = np.empty([images.shape[0], W.shape[0], W.shape[1]])  # Per-image Watermark estimation
    I_k = np.empty([images.shape[0], W.shape[0], W.shape[1]])  # Per-image Alpha estimation

    for i in range(W_k.shape[0]):
        W_k[i] = W.copy()
        I_k[i] = images[i] - W

    debug(print, "Computing derivative...")
    D_x = derivative(np.diag(alpha.ravel()).shape, 0)
    D_y = derivative(np.diag(alpha.ravel()).shape, 1)

    for itt in range(itterations):
        # 1. Image Watermark Decomposition
        for i in tqdm(range(images.shape[0]), desc="Itteration {}".format(itt + 1), leave=False):
            J = images[i]
            W_k[i], I_k[i] = image_watermark_decomposition.image_watermark_decomposition(J, I_k[i], W_k[i], W, W_hat,
                                                                                         alpha, D_x, D_y)

        # 2. Watermark Update
        W = np.median(W_k, axis=0)

        # 3. Matte Update
        alpha = matte_update.matte_update(images, I_k, W_k, W, W_hat, alpha, D_x, D_y)
    return W, alpha


def subtract_watermarks(path, images, watermark, alpha, before):
    if before:
        save_path = path + '/subtraction/before_optimization/'
    else:
        save_path = path + '/subtraction/after_optimization/'

    for i in range(images.shape[0]):
        save_image(save_path + "{0:05}.png".format(i), (images[i] - watermark * alpha) * 255)


def invert_watermark(path, images, watermark, alpha, before):
    if before:
        save_path = path + '/invertion/before_optimization/'
    else:
        save_path = path + '/invertion/after_optimization/'

    for i in range(images.shape[0]):
        save_image(save_path + "{0:05}.png".format(i), ((images[i] - watermark * alpha) / (1 - alpha)) * 255)


def remove_watermarks(images, itterations=1):
    # determine poisson estimate (initial watermark guess)
    poisson_est = initial_estimation.initial_estimation(images)
    poisson_est = normalize(poisson_est)

    # determine the matte (alpha)
    matte = single_image_matting.single_image_matting(images, poisson_est)
    # calculates blend factor (C)
    blend_factor, est_IK = blend_factor_estimation.blend_factor_estimation(images, poisson_est, matte)
    matte = matte * blend_factor

    # before optimization results for subtraction and invertion
    subtract_watermarks(save_path, images, poisson_est, matte, before=True)
    invert_watermark(save_path, images, poisson_est, matte, before=True)

    debug(show_image, matte)
    debug(show_image, poisson_est)

    images = images[0:50]
    watermark, alpha = optimization(matte, poisson_est, images, itterations)

    debug(show_image, watermark)
    debug(show_image, alpha)

    # after optimization results for subtraction and invertion
    subtract_watermarks(save_path, images, watermark, alpha, before=False)
    invert_watermark(save_path, images, watermark, alpha, before=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to a watermarked image dataset", type=str)
    args = parser.parse_args()
    return args


def mkdirs(dataset):
    dir_to_create = '../Results/' + dataset.split('/')[2] + '/' + dataset.split('/')[3]
    subdir1 = dir_to_create + '/subtraction/before_optimization'
    subdir2 = dir_to_create + '/subtraction/after_optimization'
    subdir3 = dir_to_create + '/invertion/before_optimization'
    subdir4 = dir_to_create + '/invertion/after_optimization'
    if not os.path.exists(subdir1):
        os.makedirs(subdir1)
    if not os.path.exists(subdir2):
        os.makedirs(subdir2)
    if not os.path.exists(subdir3):
        os.makedirs(subdir3)
    if not os.path.exists(subdir4):
        os.makedirs(subdir4)
    return dir_to_create


if __name__ == "__main__":
    args = parse_args()
    save_path = mkdirs(args.dataset)
    images = load_images(args.dataset)
    images = normalize(images)
    remove_watermarks(images)
