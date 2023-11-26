import os

import natsort
import pandas as pd
import numpy as np


def save_csv(img_dir="dataset/garbage_classification", ratio=None) -> np.ndarray:
    """
    클래스 별로 비율에 따라 train_data, test_data를 임의로 나누고 csv 파일로 저장
    :param img_dir: 이미지 데이터셋 경로
    :param ratio: train_data 비율, 나머지는 test_dada
    :return: 클래스별 데이터 개수 리스트
    """
    img_dir_list = natsort.natsorted(os.listdir(img_dir))

    train_name_list = np.empty(0)
    train_label_list = np.empty(0, dtype=np.int32)
    test_name_list = np.empty(0)
    test_label_list = np.empty(0, dtype=np.int32)

    class_length_list = np.empty(0, dtype=np.int32)

    for i, path in enumerate(img_dir_list):
        joined_path = os.path.join(img_dir, path)

        img_list = np.array(natsort.natsorted(os.listdir(joined_path)))
        length = len(img_list)

        idx_permute = np.random.permutation(length)  # 랜덤 인덱스

        if ratio is None:
            train_length = length - 100
        else:
            train_length = int(length * ratio)

        class_length_list = np.append(class_length_list, train_length)

        train_join_list = [os.path.join(path, img) for img in img_list[idx_permute[:train_length]]]
        train_name_list = np.append(train_name_list, train_join_list)
        train_label_list = np.append(train_label_list, [i] * train_length)

        test_join_list = [os.path.join(path, img) for img in img_list[idx_permute[train_length:]]]
        test_name_list = np.append(test_name_list, test_join_list)
        test_label_list = np.append(test_label_list, [i] * (length - train_length))

    # 클래스 별로 나누어진 리스트를 재배열
    # train_idx_permute = np.random.permutation(len(train_name_list))
    # train_name_list = train_name_list[train_idx_permute]
    # train_label_list = train_label_list[train_idx_permute]
    #
    # test_idx_permute = np.random.permutation(len(test_name_list))
    # test_name_list = test_name_list[test_idx_permute]
    # test_label_list = test_label_list[test_idx_permute]

    train = {"file_name": train_name_list,
             "label": train_label_list}

    test = {"file_name": test_name_list,
            "label": test_label_list}

    df1 = pd.DataFrame(train)
    df1.to_csv("train_data.csv", index=False)

    df2 = pd.DataFrame(test)
    df2.to_csv("test_data.csv", index=False)

    return class_length_list


def load_csv(csv_path="annotations.csv"):
    csv = pd.read_csv(csv_path)
    print(csv)


if __name__ == "__main__":
    save_csv()
