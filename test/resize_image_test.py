from PIL import Image
from project.resize_image import ResizeImage
import os
from torchvision.transforms.functional import to_pil_image

img_dir = '../project/dataset/garbage_classification/battery'
img_labels = ['battery1.jpg', 'battery2.jpg']

img_list = []
for img_label in img_labels:
    img = Image.open(os.path.join(img_dir, img_label))
    img.show()
    img_list.append(img)
    print(img.size)

resize_image = ResizeImage(resize_type='expand')
for i, img in enumerate(img_list):
    resized_img = resize_image(img)
    print("img {0}: {1}".format(i, resized_img.shape))
    pil_img = to_pil_image(resized_img)
    pil_img.show()

resize_image = ResizeImage(resize_type='crop')
for i, img in enumerate(img_list):
    resized_img = resize_image(img)
    print("img {0}: {1}".format(i, resized_img.shape))
    pil_img = to_pil_image(resized_img)
    pil_img.show()

resize_image = ResizeImage(resize_type='crush')
for i, img in enumerate(img_list):
    resized_img = resize_image(img)
    print("img {0}: {1}".format(i, resized_img.shape))
    pil_img = to_pil_image(resized_img)
    pil_img.show()

