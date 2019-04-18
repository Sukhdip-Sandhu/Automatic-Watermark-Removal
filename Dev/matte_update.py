import cv2
import numpy as np
from tqdm import tqdm
from common import *


def phi(s, epsilon=0.001):
    res = np.sqrt(s + epsilon ** 2)
    return res


def phi_prime(s, epsilon=0.001):
    res = 1 / (2 * phi(s, epsilon))
    return res


def get_gradients(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
    return np.array([gx, gy])


def matte_update(J, I_k, W_k, W_med, W_hat, alpha, D_x, D_y, lambda_alpha=0.01, beta=0.005):
    # From Supplementary Material
    debug(print, "Starting matte update...")
    alpha_x, alpha_y = get_gradients(alpha)
    alpha_diag = np.diag(alpha.ravel())

    W_diag = np.diag(W_med.ravel())

    for i in tqdm(range(J.shape[0])):
        alphaWk = alpha * W_k[i]
        alphaWk_gx, alphaWk_gy = get_gradients(alphaWk)
        phi_prime_f = np.diag(phi_prime(
            np.linalg.norm(get_gradients(alpha * W_k[i]) - get_gradients(W_hat), axis=0) ** 2).ravel())  # atten
        phi_k_a = np.diag(
            ((phi_prime(((alpha * W_k[i] + (1 - alpha) * I_k[i] - J[i]) ** 2))) * ((W_med - I_k[i]) ** 2)).ravel())
        phi_k_b = (((phi_prime(((alpha * W_k[i] + (1 - alpha) * I_k[i] - J[i]) ** 2))) * (
                    (W_med - I_k[i]) * J[i] - I_k[i])).ravel())
        phi_alpha = np.diag(phi_prime(alpha_x ** 2 + alpha_y ** 2).ravel())
        L_alpha = D_x.T.dot(phi_alpha.dot(D_x)) + D_y.T.dot(phi_alpha.dot(D_y))
        L_f = D_y.T.dot(phi_prime_f).dot(D_y) + D_x.T.dot(phi_prime_f).dot(D_x)
        A_f = W_diag.T.dot(L_f).dot(W_diag)

        if i == 0:
            A1 = phi_k_a + lambda_alpha * L_alpha + beta * A_f
            b1 = phi_k_b + beta * W_diag.dot(L_f).dot(W_hat.ravel())
        else:
            A1 += (phi_k_a + lambda_alpha * L_alpha + beta * A_f)
            b1 += (phi_k_b + beta * W_diag.T.dot(L_f).dot(W_hat.ravel()))

    x, res, rank, s = np.linalg.lstsq(A1, b1, rcond=None)
    alpha = x.reshape(J.shape[1:])

    return alpha
