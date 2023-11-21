import time

import torch
from torch.nn import CrossEntropyLoss
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from cnn import CNN
from dataloader import CustomDataset, CustomDataLoader, save_csv
from datatransform import ResizeImage, one_hot_encode
from layers import SoftmaxCrossEntropyLoss
from optimizer import *
from backward import Backward


if __name__ == "__main__":
    epochs = 10
    learning_rate = 0.00005
    train_batch_size = 32
    test_batch_size = 32

    device = "cpu"
    print(device)

    start = time.time()

    model = CNN().to(device)

    # criterion = CrossEntropyLoss().to(device)
    criterion = SoftmaxCrossEntropyLoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    load_start = time.time()
    save_csv()

    resize = ResizeImage(size=32, transform=ToTensor(), resize_type='expand')

    train_data = CustomDataset(annotations_file="train_data.csv",
                               img_dir="dataset/garbage_classification",
                               transform=resize,
                               target_transform=one_hot_encode)

    test_data = CustomDataset(annotations_file="test_data.csv",
                              img_dir="dataset/garbage_classification",
                              transform=resize,
                              target_transform=one_hot_encode)

    train_dataloader = CustomDataLoader(train_data, batch_size=train_batch_size, shuffle=True,
                                        num_workers=4, prefetch_factor=32, pin_memory=True)
    test_dataloader = CustomDataLoader(test_data, batch_size=test_batch_size, shuffle=False,
                                       num_workers=1, prefetch_factor=32, pin_memory=True)

    train_accuracy_list = []
    test_accuracy_list = []

    for epoch in range(epochs):
        print("[Epoch: {:>4}] {:0.5g} %".format(epoch + 1, 0.), end="")

        avg_cost = 0
        train_correct = 0
        train_complete = 0
        train_total = train_data.__len__()
        for i, data in enumerate(train_dataloader):
            batch_x = data[0].to(device)
            batch_y = data[1].to(device)

            optimizer.zero_grad()

            hypothesis = model(batch_x)
            cost = criterion(hypothesis, batch_y)
            # cost.backward()

            backward = Backward(model)
            backward.backward(criterion.backward())

            optimizer.step()

            avg_cost += cost.item()
            _, predicted = torch.max(hypothesis.data, 1)
            train_complete += batch_y.shape[0]
            train_correct += (predicted == torch.argmax(batch_y, dim=1)).sum().item()

            print("\r[Epoch: {:>4}] {:0.5g} %".format(epoch + 1, 100 * train_complete / train_total), end="")

        accuracy = 100 * train_correct / train_total
        train_accuracy_list.append(accuracy)
        print("\r[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, avg_cost / train_total))
        print("Train Accuracy: {0:.3f} %".format(accuracy))

        test_correct = 0
        test_complete = 0
        test_total = test_data.__len__()
        with (torch.inference_mode()):
            print("Test Progress: {:0.5g} %".format(0.), end="")
            for data in test_dataloader:
                test_x = data[0].to(device)
                test_y = data[1].to(device)

                outputs = model(test_x)

                _, predicted = torch.max(outputs.data, 1)
                test_complete += test_y.shape[0]
                test_correct += (predicted == torch.argmax(test_y, dim=1)).sum().item()

                print("\rTest Progress: {:0.5g} %".format(100 * test_complete / test_total), end="")

        test_accuracy = 100 * test_correct / test_total
        test_accuracy_list.append(test_accuracy)
        print("\rTest Accuracy: {0:.3f} %".format(100 * test_correct / test_total))

    end = time.time()
    print("총 학습 시간 : {}".format(end - start))

    plt.plot(test_accuracy_list, '-r', label="Test Accuracy")
    plt.plot(train_accuracy_list, '-b', label="Training Accuracy")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
