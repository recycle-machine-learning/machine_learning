import os
import unittest

from PIL import Image

from project.datatransform.resize_image import ResizeImage


class ResizeImageTestCase(unittest.TestCase):
    def setUp(self):
        img_dir = '../../project/dataset/garbage_classification/battery'
        img_labels = ['battery1.jpg', 'battery2.jpg']

        self.size = 128
    
        self.img_list = []
        for img_label in img_labels:
            img = Image.open(os.path.join(img_dir, img_label))
            self.img_list.append(img)

    def test_expand(self):
        resize_image = ResizeImage(size=self.size, resize_type='expand', normalize=False)
        for i, img in enumerate(self.img_list):
            resized_img = resize_image(img)

            print("img {0}: {1}".format(i, resized_img.shape))

            pil_img = Image.fromarray(resized_img)
            pil_img.show()

            self.assertEqual(pil_img.size, (self.size, self.size))

    def test_crop(self):
        resize_image = ResizeImage(size=self.size, resize_type='crop', normalize=False)
        for i, img in enumerate(self.img_list):
            resized_img = resize_image(img)

            print("img {0}: {1}".format(i, resized_img.shape))

            pil_img = Image.fromarray(resized_img)
            pil_img.show()

            self.assertEqual(pil_img.size, (self.size, self.size))
    def test_crush(self):
        resize_image = ResizeImage(size=self.size, resize_type='crush', normalize=False)
        for i, img in enumerate(self.img_list):
            resized_img = resize_image(img)

            print("img {0}: {1}".format(i, resized_img.shape))

            pil_img = Image.fromarray(resized_img)
            pil_img.show()

            self.assertEqual(pil_img.size, (self.size, self.size))
