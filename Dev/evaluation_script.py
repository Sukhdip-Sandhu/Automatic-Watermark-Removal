import argparse
import os
import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr


def load_image(image_path, color=cv2.IMREAD_GRAYSCALE):
    img = cv2.imread(image_path, color)
    return img


def load_images(directory):
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory)]
    images = np.array([load_image(i) for i in image_paths])
    return images


def calculate_PSNR(gt, rc):
    num_images = rc.shape[0]
    PSNR_values = np.zeros([num_images])

    for i in range(num_images):
        PSNR = psnr(gt[i], rc[i])
        PSNR_values[i] = PSNR
    return np.mean(PSNR_values)


def calculate_DSSIM(gt, rc):
    num_images = rc.shape[0]
    DSSIM_values = np.zeros([num_images])

    for i in range(num_images):
        DSSIM = 0.5 * (1 - ssim(gt[i], rc[i]))
        DSSIM_values[i] = DSSIM
    return np.mean(DSSIM_values)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("groundtruths", help="Path to a watermarked image dataset", type=str)
    parser.add_argument("reconstructed", help="Path to a watermarked image dataset", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    ground_truths = load_images(args.groundtruths)
    reconstructed_images = load_images(args.reconstructed)
    PSNR = calculate_PSNR(ground_truths, reconstructed_images)
    DSSIM = calculate_DSSIM(ground_truths, reconstructed_images)
    print('PSNR: \t{}\nDSSIM: \t{}'.format(PSNR, DSSIM))
