import cv2
import numpy as np

def load_image(image_path, resize_size, crop_pt_1=None, crop_pt_2=None, normalize_0_1=False):
    """
    Loads an image from a given path, crops it around the rectangle defined by two points, and resizes it.

    Inputs:
    image_path: path of the image
    resize_size: new size of the output image
    crop_pt_1: first point of crop
    crop_pt_2: second point of crop

    Returns:
    image: resized image and rescaled to [0, 1] interval
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shape = image.shape
    if crop_pt_1 is None:
        crop_pt_1 = (0, 0)
    if crop_pt_2 is None:
        crop_pt_2 = (shape[0], shape[1])
    image = image[crop_pt_1[0]:crop_pt_2[0], crop_pt_1[1]:crop_pt_2[1]]
    resized = cv2.resize(image, resize_size)
    resized = resized.astype(np.float32)

    if normalize_0_1:
        return resized / 255.0
    else:
        return (resized - 127.5) / 127.5