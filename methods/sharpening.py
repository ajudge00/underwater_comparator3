import cv2
import numpy as np
from scipy.ndimage.filters import median_filter
from .myutils import convert_dtype, ConvertFlags


def num2(img: np.ndarray):
    gaussian_filtered = cv2.GaussianBlur(img, (15, 15), 0)
    diff = cv2.subtract(img, gaussian_filtered)

    min_val = np.min(diff)
    max_val = np.max(diff)

    stretched_img = (diff - min_val) / (max_val - min_val)

    return (img + stretched_img) / 2


def norm_unsharp_mask_matlab(img: np.ndarray, sigma=20, N=30, gain=1.0):
    img = img.astype(np.float32) / 255.0

    Igauss = img.copy()

    for _ in range(N):
        Igauss = cv2.GaussianBlur(Igauss, (0, 0), sigma)
        Igauss = np.minimum(img, Igauss)

    Norm = img - gain * Igauss

    Norm_eq = np.zeros_like(Norm)
    for n in range(3):
        Norm_eq[:, :, n] = cv2.equalizeHist((Norm[:, :, n] * 255).astype(np.uint8)) / 255.0

    Isharp = (img + Norm_eq) / 2
    return (Isharp * 255).astype(np.uint8)


def norm_unsharp_mask(img: np.ndarray):
    """
    Normalized unsharp mask\n
    S = (I + N {I − G ∗ I}) / 2\n
    where\n
    - G ∗ I is the Gaussian-filtered image and\n
    - N is the linear normalization (histogram stretching) operator.

    :param img: image to process
    :return: S
    """

    assert img.ndim == 3 and img.dtype == np.uint8 and 1 <= np.max(img) <= 255

    gaussian_filtered = cv2.GaussianBlur(img, (15, 15), 0)  # Experiment with kernel size

    # Step 2: High-pass signal
    diff = cv2.subtract(img, gaussian_filtered)  # Difference I - G ∗ I

    # Step 3: Normalize the difference (histogram stretching)
    min_val, max_val = diff.min(), diff.max()
    diff = ((diff - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)

    # Step 4: Combine with original image and average
    sharpened = ((img.astype(np.float32) + diff.astype(np.float32)) / 2.0).astype(np.uint8)

    return sharpened
