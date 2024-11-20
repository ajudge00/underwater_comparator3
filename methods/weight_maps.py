import enum

import cv2
import numpy as np


class WeightMapMethods(enum.Enum):
    LAPLACIAN = 0,
    SATURATION = 1,
    SALIENCY = 2,
    SALIENCY2 = 3


def laplacian_contrast_weight(img: np.ndarray):
    # kb jó, a matlabos máshogy néz ki, de ez a cikkben lévőre hasonlít

    assert img.ndim == 3 and img.dtype == np.uint8 and 1 <= np.max(img) <= 255

    laplacian_edges = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), ddepth=cv2.CV_32F, ksize=7)
    laplacian_edges = np.abs(cv2.normalize(laplacian_edges, None, -255, 255, cv2.NORM_MINMAX, cv2.CV_32F))

    laplacian_base = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_8U, ksize=3)
    laplacian_base = cv2.normalize(laplacian_base, None, 0, 255, cv2.NORM_MINMAX)

    contrast_weight = 0.4 * laplacian_base + 0.6 * laplacian_edges

    return 255 - contrast_weight.astype(np.uint8)


def saliency_weight(img: np.ndarray):
    # this is rly fckin cool but doesnt work properly
    assert img.ndim == 3 and img.dtype == np.uint8 and 1 <= np.max(img) <= 255

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    mean_image_feature_vector = img_lab.mean(axis=(0, 1))

    binomial_kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    binomial_kernel_1d /= binomial_kernel_1d.sum()
    img_blurred = cv2.sepFilter2D(img_lab, -1, binomial_kernel_1d, binomial_kernel_1d)

    diff = cv2.subtract(img_blurred, mean_image_feature_vector)
    W_Sal = np.sqrt(np.sum(diff ** 2, axis=2))

    # return cv2.normalize(W_Sal, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    return cv2.normalize(W_Sal.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def saturation_weight(img: np.ndarray):
    """
    W_Sat = √( (1/3) * [(R_k − L_k )^2 +(G_k − L_k )^2 +(B_k − L_k )^2])
    :param img: BGR input image (float32)
    :return: Saturation weight map
    """
    assert img.ndim == 3 and img.dtype == np.uint8 and 1 <= np.max(img) <= 255

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L = img_lab[:, :, 0] / 255.0

    B, G, R = cv2.split(img)

    R = R.astype(np.float32)
    G = G.astype(np.float32)
    B = B.astype(np.float32)

    W_Sat = cv2.sqrt(
        (1 / 3) * ((R - L) ** 2 + (G - L) ** 2 + (B - L) ** 2)
    )

    W_Sat = cv2.normalize(W_Sat, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)

    return W_Sat


def get_weight_map(img: np.ndarray, method: WeightMapMethods) -> np.ndarray:
    if method == WeightMapMethods.LAPLACIAN:
        return laplacian_contrast_weight(img)
    elif method == WeightMapMethods.SATURATION:
        return saturation_weight(img)
    elif method == WeightMapMethods.SALIENCY:
        return saliency_weight(img)
    elif method == WeightMapMethods.SALIENCY2:
        return get_saliency_ft(img)


def normalize_weight_maps(
        lap1: np.ndarray, sal1: np.ndarray, sat1: np.ndarray,
        lap2: np.ndarray, sal2: np.ndarray, sat2: np.ndarray,
        reg_term=0.1) -> tuple[np.ndarray, np.ndarray]:
    denom = lap1 + sal1 + sat1 + lap2 + sal2 + sat2 + 2 * reg_term
    W_Normalized1 = (lap1 + sal1 + sat1 + reg_term) / denom
    W_Normalized2 = (lap2 + sal2 + sat2 + reg_term) / denom

    W_Normalized1 = cv2.normalize(W_Normalized1, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    W_Normalized2 = cv2.normalize(W_Normalized2, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)

    return W_Normalized1, W_Normalized2


def get_saliency_ft(img: np.ndarray):
    mean_val = np.mean(img, axis=(0, 1))

    im_blurred = cv2.GaussianBlur(img, (5, 5), 0)

    sal = np.linalg.norm(mean_val - im_blurred, axis=2)
    sal_max = np.max(sal)
    sal_min = np.min(sal)
    sal = ((sal - sal_min) / (sal_max - sal_min))

    sal = cv2.normalize(sal, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    return sal
