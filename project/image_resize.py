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


def keep_ratio_resize_dataset_image(path, target_height=128, target_width=128):
    image_list = os.listdir(path)
    image_ary = []
    for image in image_list:
        img = Image.open(path + '/' + image)

        width, height = img.size
        length = max(width, height)
        x = (width - length) // 2
        y = (height - length) // 2
        sliced_img = img.crop((x, y, x + length, y + length))
        resized_img = sliced_img.resize((target_height, target_width))

        img_np = np.array(resized_img)
        image_ary.append(img_np)
    return np.array(image_ary)