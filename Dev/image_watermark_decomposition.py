import numpy as np
import numpy.linalg
from common import *
import time
import cv2


# The "Derivative Operator"
# Adapted from:
# https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size
# Changed to allow even sized kernels to support the algorithm
def derivative(shape, axis):
    """
    shape must be odd: eg. (5,5)
    axis is the direction, with 0 to positive x and 1 to positive y
    """
    # Closest Higher Odd
    odd_y = False
    odd_x = False
    if shape[0] % 2 == 0:
        odd_x = True
        shape = (int((shape[0] / 2.) + 0.1) * 2 + 1, shape[1])
    if shape[1] % 2 == 0:
        odd_y = True
        shape = (shape[0], int((shape[1] / 2.) + 0.1) * 2 + 1)

    k = np.zeros(shape)

    p = [(j, i) for j in range(shape[0])
         for i in range(shape[1])
         if not (i == (shape[1] - 1) / 2. and j == (shape[0] - 1) / 2.)]

    for j, i in p:
        j_ = int(j - (shape[0] - 1) / 2.)
        i_ = int(i - (shape[1] - 1) / 2.)
        k[j, i] = (i_ if axis == 0 else j_) / float(i_ * i_ + j_ * j_)

    if odd_x:
        k = np.delete(k, int(k.shape[0] / 2), 0)

    if odd_y:
        k = np.delete(k, int(k.shape[1] / 2), 1)

    k = k.astype(np.float32)
    return k


def _phi(s, epsilon=0.001):
    res = np.sqrt(s + epsilon ** 2)
    return res


def _phi_prime(s, epsilon=0.001):
    res = 1 / (2 * _phi(s, epsilon))
    return res


def get_gradients(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
    return np.array([gx, gy])


# Input:
#   J: Watermarked image
#   I: The estimated I specific to the image
#   W_k: The image specific estimated watermark
#   W: Estimated watermark that changes with itterations
#   W_m: The initial estimated watermark from the previous step
#   alpha: The estimated alpha matte of the image
#   lambda_I: [0, 1]
#   lambda_w: [0.001, 0.1]
#   lambda_alpha: 0.01
#   beta: [0.001, 0.01]
#   gamma: ?????? who knows!!!!!!!
# Output:
#   M and b where M * [W, I] = b for irls optimization
def _decomp(J, I_k, W_k, W, W_m, alpha, D_x, D_y, lambda_I=1, lambda_w=0.005, lambda_alpha=0.01, beta=1, gamma=1):
    # From Supplementary Material
    # The original image estimate is the watermarked image minus the watermark
    I_kx, I_ky = get_gradients(I_k)
    W_x, W_y = get_gradients(W)
    W_kx, W_ky = get_gradients(W_k)
    alpha_x, alpha_y = get_gradients(alpha)
    abs_alpha_x = np.absolute(alpha_x)
    abs_alpha_y = np.absolute(alpha_y)
    alpha_diag = np.diag(alpha.ravel())
    alpha_diag_bar = 1 - alpha_diag
    # Error between The actual watermarked image and the estimated watermarked image
    phi_prime_data = np.diag(_phi_prime((alpha * W_k + (1 - alpha) * I_k - J) ** 2).ravel())

    phi_prime_W = np.diag(_phi_prime( ( abs_alpha_x * W_kx + abs_alpha_y * W_ky )**2 ).ravel() )
    phi_prime_I = np.diag(_phi_prime( ( abs_alpha_x * I_kx + abs_alpha_y * I_ky )**2 ).ravel() )

    # Error between the initial watermark guess and the current guess, used to keep them "close"
    # Here there is a mistake in the supplementary material, brackets
    phi_prime_f = np.diag(
        _phi_prime(numpy.linalg.norm(get_gradients(alpha * W_k) - get_gradients(W_m), axis=0) ** 2).ravel())

    phi_prime_aux = np.diag(_phi_prime((W_k - W) ** 2).reshape(-1))
    phi_prime_rI = np.diag(_phi_prime(abs_alpha_x * I_kx ** 2 + abs_alpha_y * I_ky ** 2).ravel())
    phi_prime_rW = np.diag(_phi_prime(abs_alpha_x * W_x ** 2 + abs_alpha_y * W_y ** 2).ravel())

    c_x = np.diag(abs_alpha_x.ravel())
    c_y = np.diag(abs_alpha_y.ravel())

    L_f = D_y.T.dot(phi_prime_f).dot(D_y) + D_x.T.dot(phi_prime_f).dot(D_x)
    L_I = D_x.T.dot(c_x * phi_prime_rI).dot(D_x) + D_y.T.dot(c_y * phi_prime_rI).dot(D_y)
    L_w = D_x.T.dot(c_x * phi_prime_rW).dot(D_x) + D_y.T.dot(c_y * phi_prime_rW).dot(D_y)
    A_f = alpha_diag.T.dot(L_f * alpha_diag) + gamma * phi_prime_aux

    b_w = alpha_diag.T.dot(phi_prime_data).dot(J.ravel()) + beta * L_f.dot(W_m.ravel()) + gamma * phi_prime_aux.dot(
        W.ravel())
    b_I = alpha_diag_bar.T.dot(phi_prime_data).dot(J.ravel())

    b = np.vstack((b_w, b_I)).reshape(W.shape[0] ** 2 * 2, 1)

    upper_left = (alpha_diag ** 2).dot(phi_prime_data) + lambda_w * L_w + beta * A_f
    upper_right = alpha_diag.dot(alpha_diag_bar).dot(phi_prime_data)  # Bottleneck
    bottom_left = upper_right
    bottom_right = (alpha_diag_bar ** 2).dot(phi_prime_data) + lambda_I * L_I

    M = np.vstack((np.hstack((upper_left, upper_right)), np.hstack((bottom_left, bottom_right))))

    return M, b


# Input:
#   J: Image with the watermark
#   I_k: The previously estimated alpha
#   W_k: The previously estimated watermark
#   W_hat: Original estimate
#   alpha: Global alpha estimate
# Output:
#   W: The watermark estimate
#   I: The image estimate estimate
def image_watermark_decomposition(J, I_k, W_k, W, W_hat, alpha, D_x, D_y):
    debug(print, "Starting Image-Watermark Decomposition...")
    t1 = time.time()
    M, b = _decomp(J, I_k, W_k, W, W_hat, alpha, D_x, D_y)
    debug(print, "Decomposition Time: {}".format(time.time() - t1))

    t1 = time.time()
    x, res, rank, s = numpy.linalg.lstsq(M, b, rcond=None)
    debug(print, "Least Square Regressor Time: {}".format(time.time() - t1))

    W = x[:J.shape[0] * J.shape[1]]
    I = x[J.shape[0] * J.shape[1]:]
    W = W.reshape(J.shape)
    I = I.reshape(J.shape)
    debug(cv2.imwrite, "test_I.png", I)
    debug(cv2.imwrite, "test_W.png", W)
    return W, I


if __name__ == "__main__":
    pass
