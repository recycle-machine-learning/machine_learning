import os
import multiprocessing as mp

import numpy as np
from PIL import Image

from ..datatransform.resize_image import ResizeImage


def load_data(size=64, normalize=True):
    listdir = os.listdir('dataset/garbage_classification')
    # .DS_Store 디렉토리 제외
    file_list = [file for file in listdir if not file.startswith('.DS_Store')]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        print(file_list)
        pool_result = pool.starmap(load_data_single_class,
                                   zip(file_list,
                                       [idx for idx in range(len(file_list))],
                                       [size] * len(file_list),
                                       [normalize] * len(file_list)))

    x_train = [result[0] for result in pool_result]
    y_train = [result[1] for result in pool_result]
    x_test = [result[2] for result in pool_result]
    y_test = [result[3] for result in pool_result]

    x_train = np.concatenate(x_train, axis=0)  # data_num x 3 x size x size
    y_train = np.concatenate(y_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    train_random_index = np.random.permutation(len(x_train))
    test_random_index = np.random.permutation(len(x_test))

    x_train = x_train[train_random_index]
    y_train = y_train[train_random_index]
    x_test = x_test[test_random_index]
    y_test = y_test[test_random_index]

    return x_train, y_train, x_test, y_test


def load_data_single_class(path, class_idx, size=64, normalize=True):
    path = 'dataset/garbage_classification/' + path

    image_ary = resize_image(path, size, normalize)

    image_file = np.array(image_ary)
    image_file = image_file.transpose(0, 3, 1, 2)

    target = np.array([class_idx] * len(image_file))
    print(class_idx, image_file.shape, target.shape)

    train_ratio = 0.9
    train_len = int(len(image_file) * train_ratio)

    x_train = image_file[:train_len]
    y_train = target[:train_len]
    x_test = image_file[train_len:]
    y_test = target[train_len:]

    return x_train, y_train, x_test, y_test


def resize_image(path, size, normalize):
    image_list = os.listdir(path)
    image_ary = []

    for image in image_list:
        image = Image.open(path + '/' + image).convert('RGB')

        resize = ResizeImage(size=size, resize_type='expand', normalize=normalize)
        img_resized = resize(image)
        image_ary.append(img_resized)

    return np.array(image_ary)
