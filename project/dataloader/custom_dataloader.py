import torch.utils.data as data


class CustomDataLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(CustomDataLoader, self).__init__(*args, **kwargs)

# def _collate_fn(batch):
#     batch = filter(lambda x: x is not None, batch)
#     images, labels = zip(*batch)
#
#     transform = ResizeImage(size=64, transform=ToTensor(), resize_type='expand')
#
#     resized_images = []
#     for img in images:
#         transformed_img = transform(img)
#         resized_images.append(transformed_img)
#
#     img_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
#     label_tensors = torch.cat([t.unsqueeze(0) for t in labels], 0)
#
#     print(images.shape)
#     print(labels.shape)

    # return images, labels
