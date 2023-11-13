import os

import numpy as np
from PIL import Image

from project.datatransform.resize_image import ResizeImage


def resize_dataset_image(path, target_height=128, target_width=128):
    image_list = os.listdir(path)
    image_ary = []

    for image in image_list:
        # img = Image.open(path + '/' + image).convert('RGB').resize((target_height, target_width))
        # img_np = np.array(img)

        resize = ResizeImage(size=64, resize_type='expand')
        img_resized = resize(image)
        image_ary.append(img_resized)

        # if len(image_list) < 5000:
        #     if len(image_ary) < 2000:
        #         img2 = img.rotate(90)   # 회전
        #         img3 = img.transpose(Image.FLIP_LEFT_RIGHT)  # 좌우반전
        #         img4 = img.transpose(Image.FLIP_TOP_BOTTOM)  # 상하반전
        #         img5 = img.rotate(180)
        #         img6 = img.rotate(250)
        #         img2_np = np.array(img2)
        #         img3_np = np.array(img3)
        #         img4_np = np.array(img4)
        #         img5_np = np.array(img5)
        #         img6_np = np.array(img6)
        #         image_ary.append(img2_np)
        #         image_ary.append(img3_np)
        #         image_ary.append(img4_np)
        #         image_ary.append(img5_np)
        #         image_ary.append(img6_np)

    return np.array(image_ary)


def keep_ratio_resize_dataset_image(path, target_height=128, target_width=128):
    image_list = os.listdir(path)
    image_ary = []
    for image in image_list:
        img = Image.open(path + '/' + image).convert('RGB')

        width, height = img.size
        length = max(width, height)
        x = (width - length) // 2
        y = (height - length) // 2
        sliced_img = img.crop((x, y, x + length, y + length))
        resized_img = sliced_img.resize((target_height, target_width))

        img_np = np.array(resized_img)
        image_ary.append(img_np)
    return np.array(image_ary)
