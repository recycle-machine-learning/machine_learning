import os
import numpy as np
from PIL import Image


def resize_dataset_image(path, target_height=128, target_width=128):
    image_list = os.listdir(path)
    image_ary = []
    for image in image_list:
        img = Image.open(path + '/' + image).resize((target_height, target_width))
        img_np = np.array(img)
        image_ary.append(img_np)
    return np.array(image_ary)


def keep_ratio_resize_dataset_image(path):
    image_list = os.listdir(path)
    image_ary = []
    for image in image_list:
        img = Image.open(path + '/' + image)
        width, height = img.size
        half_min_length = min(width, height) // 2
        # img.crop((width // 2, height // 2, width // 2, height // 2))
        img_np = np.array(img)
        image_ary.append(img_np)
    return np.array(image_ary)
