import numpy as np
from PIL import Image


class ResizeImage(object):
    def __init__(self, channel=3, size=64, transform=None, resize_type='expand', normalize=True):
        self.transform = transform
        self.channel = channel
        self.size = size
        self.resize_type = resize_type
        self.normalize = normalize

    def __call__(self, img):
        if self.resize_type == 'expand':
            img = self.pad_expand(img)
        elif self.resize_type == 'crop':
            img = self.center_crop(img)
        elif self.resize_type == 'crush':
            img = self.crush_resize(img)

        if self.normalize:
            img_numpy = img.astype(np.float64) / 255

        if self.transform:
            return self.transform(img)
        return img

    def pad_expand(self, img: Image.Image) -> np.ndarray:
        # 이미지의 긴 부분을 self.size에 맞춤
        img.thumbnail((self.size, self.size))
        w, h = img.size

        img_padding = Image.new("RGB", (self.size, self.size), color=0)
        img_padding.paste(img, (0, 0))
        img_padding = np.array(img_padding)
        print(img_padding.shape)

        img = np.array(img)

        # 남는 공간을 마지막 행(열)을 확장시켜 채움
        if w > h:
            img_padding[h:] = img[h - 1]
        else:
            img_padding[:, w:] = img[:, w - 1, np.newaxis]

        return img_padding

    def center_crop(self, img: Image.Image) -> np.ndarray:
        # 가운데 정사각형 공간만큼 맞춰 자름
        w, h = img.size

        if w > h:
            img.thumbnail((w, self.size))
        else:
            img.thumbnail((self.size, h))

        w, h = img.size
        left = (w - self.size) // 2
        top = (h - self.size) // 2
        right = left + self.size
        bottom = top + self.size

        img_cropped = img.crop((left, top, right, bottom))
        return np.array(img_cropped)

    def crush_resize(self, img: Image.Image) -> np.ndarray:
        # 비율 상관없이 찌그러뜨림
        img_resized = img.resize((self.size, self.size))
        return np.array(img_resized)
