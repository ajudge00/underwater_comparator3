import cv2

from methods.myutils import *


class CompChannel(enum.Enum):
    COMP_RED = 0
    COMP_BLUE = 1


def comp_for_channel(channel: CompChannel, img: np.ndarray, alpha=1):
    """
    I_rc(x) = I_r(x) + α * (¯I_g − ¯I_r) * (1 − I_r(x)) * I_g(x)\n
    I_bc(x) = I_b(x) + α * (¯I_g − ¯I_b) * (1 − I_b(x)) * I_g(x)

    :param channel: The channel to compensate for (COMP_RED or COMP_BLUE)
    :param img: The image to be processed (has to be [0.0-1.0] float32)
    :param alpha: The strength of the compensation
    """

    assert img.ndim == 3 and img.dtype == np.uint8 and 1 <= np.max(img) <= 255

    img_f32 = convert_dtype(img, ConvertFlags.UINT8_TO_F32)

    avg_b = np.mean(img_f32[:, :, 0])
    avg_g = np.mean(img_f32[:, :, 1])
    avg_r = np.mean(img_f32[:, :, 2])

    res = img_f32.copy()

    if channel == CompChannel.COMP_RED:
        res[:, :, 2] = img_f32[:, :, 2] + alpha * (avg_g - avg_r) * (1 - img_f32[:, :, 2]) * img_f32[:, :, 1]
    elif channel == CompChannel.COMP_BLUE:
        res[:, :, 0] = img_f32[:, :, 0] + alpha * (avg_g - avg_b) * (1 - img_f32[:, :, 0]) * img_f32[:, :, 1]

    res = convert_dtype(res, ConvertFlags.F32_TO_UINT8)

    return res


def gray_world(img: np.ndarray):
    assert img.ndim == 3 and img.dtype == np.uint8 and 1 <= np.max(img) <= 255

    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])

    # Tanulság: ez hülyeség volt, mert ilyen képeknél *mindig* a zöld
    # lesz a domináns csatorna, azaz alpha és beta *mindig* nagyobb lesz 1.0-nál
    # és a clippeléssel így mindig 1.0-t kapunk, ami 1-gyel szorzás lesz :/
    # alpha = min(1.0, avg_g / avg_r)
    # betha = min(1.0, avg_g / avg_b)

    alpha = avg_g / avg_r
    betha = avg_g / avg_b

    res = img.copy()

    res[:, :, 2] = cv2.multiply(res[:, :, 2], alpha)
    res[:, :, 0] = cv2.multiply(res[:, :, 0], betha)

    return res
