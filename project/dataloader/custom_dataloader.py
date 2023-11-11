import torch
import torch.utils.data as data

from project.datatransform.resize_image import ResizeImage


class CustomDataLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(CustomDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


def _collate_fn(batch):
    batch = filter(lambda x: x is not None, batch)
    images, labels = zip(*batch)

    transform = ResizeImage(resize_type='expand')

    resized_images = []
    for img in images:
        transformed_img = transform(img)
        resized_images.append(transformed_img)

    img_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

    return img_tensors, labels
