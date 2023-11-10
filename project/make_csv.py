import os
import natsort
import pandas as pd


def save_csv(dataset_dir='dataset/garbage_classification'):
    img_dir = 'dataset/garbage_classification'
    img_dir_list = natsort.natsorted(os.listdir(img_dir))

    train_name_list = []
    train_label_list = []

    test_name_list = []
    test_label_list = []

    for i, path in enumerate(img_dir_list):
        joined_path = os.path.join(img_dir, path)

        img_list = natsort.natsorted(os.listdir(joined_path))

        train_join_list = [os.path.join(path, img) for img in img_list[:-10]]
        train_name_list.extend(train_join_list)
        train_label_list.extend([i] * (len(img_list) - 10))

        test_join_list = [os.path.join(path, img) for img in img_list[-10:]]
        test_name_list.extend(test_join_list)
        test_label_list.extend([i] * 10)

    train = {'file_name': train_name_list,
             'label': train_label_list}

    test = {'file_name': test_name_list,
            'label': test_label_list}

    df1 = pd.DataFrame(train)
    df1.to_csv('train_data.csv', index=False)

    df2 = pd.DataFrame(test)
    df2.to_csv('test_data.csv', index=False)


def load_csv(csv_path='annotations.csv'):
    csv = pd.read_csv(csv_path)
    print(csv)


if __name__ == '__main__':
    save_csv()


