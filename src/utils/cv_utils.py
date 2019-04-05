import cv2
import numpy as np

eps = 0.0001


def to_colormap_image(mat, color_map=cv2.COLORMAP_JET):
    mat -= np.mean(mat)
    mat /= max(eps, np.max(np.abs(mat)))
    mat += 1.0
    mat *= (255.0 / 2)
    mat = np.array(mat, dtype=np.uint8)
    im = cv2.applyColorMap(mat, color_map)
    return im
