import os
import image_resize
import numpy as np
import multiprocessing as mp

def load_data():
    listdir = os.listdir('./project/dataset/garbage_classification')
    file_list = [file for file in listdir if not file.startswith('.DS_Store')]
    garbage_class = file_list
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        print(file_list)
        pool_result = pool.starmap(load_data_single_class, zip(file_list, [garbage_class.index(file) for file in file_list]))
    for train, test in pool_result:
        x_train.append(train[0])
        y_train.append(train[1])
        x_test.append(test[0])
        y_test.append(test[1])
    x_train = np.concatenate(x_train, axis=0) # 12 x 10000 x 3 x 128 x 128
    y_train = np.concatenate(y_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    train_random_index = np.random.permutation(len(x_train))
    test_random_index = np.random.permutation(len(x_test))
    x_train = np.take_along_axis(x_train, train_random_index[:,np.newaxis,np.newaxis,np.newaxis], axis=0)
    y_train = np.take_along_axis(y_train, train_random_index, axis=0)
    x_test = np.take_along_axis(x_test, test_random_index[:,np.newaxis,np.newaxis,np.newaxis], axis=0)
    y_test = np.take_along_axis(y_test, test_random_index, axis=0)

    return x_train, y_train, x_test, y_test

def load_data_single_class(path,class_idx):
    image_file = image_resize.resize_dataset_image(path='./project/dataset/garbage_classification/' + path, target_height=64, target_width=64)
    image_file = np.reshape(image_file, (-1, 3, 64, 64))
    # image_file = image_file.astype('float32') / 255
    target = np.array([class_idx] * len(image_file))
    print(class_idx,image_file.shape, target.shape)
    train_ratio = 0.9
    train_len = int(len(image_file) * train_ratio)
    x_train = image_file[:train_len]
    y_train = target[:train_len]
    x_test = image_file[train_len:]
    y_test = target[train_len:]
    return (x_train,y_train),(x_test,y_test)

