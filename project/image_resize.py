import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
def resize_dataset_image(path, target_height=128, target_width=128):
    image_list = os.listdir(path)
    image_ary = []
    for image in image_list:
        img = Image.open(path + '/' + image).convert('RGB').resize((target_height, target_width))
        img_np = np.array(img)
        image_ary.append(img_np)
    return np.array(image_ary)

#train이랑 test 데이터 추출
def load_data(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3,stratify=y)
    return x_train,x_test,y_train,y_test
def labeling(path,idx,label):
    data_num = len(os.listdir(path))
    for i in range(data_num):
        label.append(idx)


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