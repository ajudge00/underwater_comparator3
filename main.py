import os

import cv2
from methods.wb_comps import *
from methods.gamma_comps import *
from methods.weight_maps import *

# from methods.sharpening import *
# from methods.fusion import *

img_original = cv2.imread('images/buvar.jpg', cv2.IMREAD_COLOR)

"""
White-Balance Pre-comp for red (and blue)
"""
img_wb_precomp = comp_for_channel(CompChannel.COMP_RED, img_original, alpha=1.0)
img_wb_precomp = comp_for_channel(CompChannel.COMP_BLUE, img_wb_precomp, alpha=1.0)

"""
White-Balance Correction using the Gray-World Assumption
"""
img_wb = gray_world(img_wb_precomp)

"""
Input 1 (Gamma Correction) and its weight maps for the eventual Multi-Scale Fusion process
"""
input1 = gamma_correction(img_wb, 2.5)
input1_laplacian_wm = get_weight_map(input1, method=WeightMapMethods.LAPLACIAN)
input1_saliency_wm = get_weight_map(input1, method=WeightMapMethods.SALIENCY)
# input1_saturation_wm = get_weight_map(input1, method=WeightMapMethods.SATURATION)

"""
Input 2 (Normalized Unsharp Masking) and its weight maps for the eventual Multi-Scale Fusion process
"""
# input2 = (img_wb)
# input2_laplacian_wm = get_weight_map(input2, method=WeightMapMethods.LAPLACIAN)
# input2_saliency_wm = get_weight_map(input2, method=WeightMapMethods.SALIENCY)
# input2_saturation_wm = get_weight_map(input2, method=WeightMapMethods.SATURATION)

"""
Normalized weight maps of Input1 and Input2
"""
# input1_norm_wm, input2_norm_wm = normalize_weight_maps(
#     input1_laplacian_wm, input1_saliency_wm, input1_saturation_wm,
#     input2_laplacian_wm, input2_saliency_wm, input2_saturation_wm
# )

"""
Naive Fusion
"""
# input1_norm_wm = cv2.cvtColor(input1_norm_wm, cv2.COLOR_GRAY2BGR)
# input2_norm_wm = cv2.cvtColor(input2_norm_wm, cv2.COLOR_GRAY2BGR)
# R_naive = apply_fusion(
#     FusionType.NAIVE,
#     input1, input2,
#     input1_norm_wm, input2_norm_wm
# )

"""
Multi-Scale Fusion
"""
# R_msf = apply_fusion(
#     FusionType.MULTI_SCALE,
#     input1, input2,
#     cv2.cvtColor(input1_norm_wm, cv2.COLOR_BGR2GRAY), cv2.cvtColor(input2_norm_wm, cv2.COLOR_BGR2GRAY),
#     levels=10
# )

"""
Images to display
"""
# cv2.imshow('Initial', img_original)
# cv2.imshow('After Pre-comp', img_wb_precomp)
# cv2.imshow('After White Balance (Gray World)', img_wb)

# cv2.imshow('Input1 (Gamma)', input1)
# cv2.imshow('Laplacian Contrast Weight (Input1)', input1_laplacian_wm)
cv2.imshow('Saliency Weight (Input1)', input1_saliency_wm)
cv2.imwrite("images/results/sal.jpg", input1_saliency_wm)
# cv2.imshow('Saturation Weight (Input1)', input1_saturation_wm)

# cv2.imshow('Input2 (Normalized Unsharp Masking)', input2)
# cv2.imshow('Laplacian Contrast Weight (Input2)', input2_laplacian_wm)
# cv2.imshow('Saliency Weight (Input2)', input2_saliency_wm)
# cv2.imshow('Saturation Weight (Input2)', input2_saturation_wm)

# cv2.imshow('Normalized Weight Map of Input1', input1_norm_wm)
# cv2.imshow('Normalized Weight Map of Input2', input2_norm_wm)

# cv2.imshow('Naive Fusion', R_naive)
# cv2.imshow('Multi-Scale Fusion', R_msf)

# filenames = {
    # 'original': img_original,
    # 'wb_precomp': img_wb_precomp,
    # 'wb': img_wb,
    # 'input1': input1,
    # 'input1_laplacian': input1_laplacian_wm,
    # 'input1_saliency': input1_saliency_wm,
    # 'input1_saturation': input1_saturation_wm,
    # 'input1_normalized': input1_norm_wm,
    # 'input2': input2,
    # 'input2_laplacian': input2_laplacian_wm,
    # 'input2_saliency': input2_saliency_wm,
    # 'input2_saturation': input2_saturation_wm,
    # 'input2_normalized': input2_norm_wm,
    # 'R_naive': R_naive,
    # 'R_msf': R_msf
# }

# if not os.path.exists('images/results'):
#     os.mkdir('images/results')

# i = 0
# for filename, image in filenames.items():
#     # image = cv2.putText(image, filename, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     image = (image * 255).astype(np.uint8)
#     path = 'results/' + "{:02d}".format(i) + f'_{filename}.jpg'

#     cv2.imwrite(path, image)
#     i += 1

cv2.waitKey(0)
cv2.destroyAllWindows()
