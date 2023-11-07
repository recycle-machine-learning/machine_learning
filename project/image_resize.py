import os
import numpy as np
from PIL import Image


def resize_dataset_image(path, target_height=200, target_width=200):
    image_list = os.listdir(path)
    image_ary = []
    for image in image_list:
        img = Image.open(path + '/' + image).resize((target_height, target_width))
        img_np = np.array(img)
        image_ary.append(img_np)
    return np.array(image_ary)
