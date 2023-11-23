import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        """
        :param annotations_file: 이미지 정보를 저장한 csv 파일
        :param img_dir: 이미지 경로
        :param transform: 전달받은 함수로 이미지를 변환
        :param target_transform: 전달받은 함수로 label을 변환
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        idx 번째 이미지 하나를 RGB 이미지로 반환
        :param idx: 이미지의 순서 번호
        :return: 이미지 1개
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
