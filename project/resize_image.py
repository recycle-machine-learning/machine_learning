import torch
import torchvision.transforms as transforms
from torchvision.transforms import Resize


class ResizeImage(object):
    def __init__(self, channel=3, size=128, resize_type='expand'):
        self.toTensor = transforms.ToTensor()
        self.channel = channel
        self.size = size
        self.resize_type = resize_type

    def __call__(self, img):
        if self.resize_type == 'expand':
            return self.pad_expand(img)
        elif self.resize_type == 'crop':
            return self.center_crop(img)
        elif self.resize_type == 'crush':
            return self.crush_resize(img)

    def pad_expand(self, img):
        w, h = img.size
        # 이미지의 긴 부분을 self.size로 맞춤
        if w > h:
            resize = Resize((int(self.size * h / w), self.size))
        else:
            resize = Resize((self.size, int(self.size * w / h)))

        resized_img = resize(img)
        w, h = resized_img.size
        tensor_img = self.toTensor(resized_img)

        pad_img = torch.FloatTensor(self.channel, self.size, self.size).fill_(0)
        # 남는 공간을 마지막 행(열)을 확장시켜 채움
        if w > h:
            pad_img[:, :h, :] = tensor_img  # bottom pad
            pad_img[:, h:, :] = tensor_img[:, h - 1, :].unsqueeze(1).expand(self.channel, self.size - h, w)
        else:
            pad_img[:, :, :w] = tensor_img  # right pad
            pad_img[:, :, w:] = tensor_img[:, :, w - 1].unsqueeze(2).expand(self.channel, h, self.size - w)

        return pad_img

    def center_crop(self, img):
        # 가운데 정사각형 공간만큼 맞춰 자름
        crop = transforms.CenterCrop(self.size)
        cropped_img = crop(img)
        return self.toTensor(cropped_img)

    def crush_resize(self, img):
        # 비율 상관없이 찌그러뜨림
        resize = Resize((self.size, self.size))
        resized_img = resize(img)
        return self.toTensor(resized_img)
        