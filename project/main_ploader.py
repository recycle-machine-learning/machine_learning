import time

from torch.utils.data import WeightedRandomSampler, RandomSampler
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
from cnn_pload import CNN_pload
from dataloader import CustomDataset, CustomDataLoader, save_csv
from util import ResizeImage, one_hot_encode, make_weights
from layers import SoftmaxCrossEntropyLoss
from optimizer import *
from backward import Backward

with torch.no_grad():
    if __name__ == "__main__":
        epochs = 1
        learning_rate = 0.00005
        train_batch_size = 32
        test_batch_size = 32
        size = 32
        out_channel1 = 16
        out_channel2 = 32
        isWeighted = True

        device = "cpu"
        print(device)

        start = time.time()
        model = CNN_pload(size, out_channel1, out_channel2).to(device)
        criterion = SoftmaxCrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        load_start = time.time()
        class_length = save_csv()

        resize = ResizeImage(size=size, transform=ToTensor(), resize_type='expand')

        train_data = CustomDataset(annotations_file="train_data.csv",
                                   img_dir="dataset/garbage_classification",
                                   transform=resize,
                                   target_transform=one_hot_encode)

        test_data = CustomDataset(annotations_file="test_data.csv",
                                  img_dir="dataset/garbage_classification",
                                  transform=resize,
                                  target_transform=one_hot_encode)

        if isWeighted:
            weights = make_weights(class_length)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        else:
            sampler = RandomSampler(train_data)

        train_dataloader = CustomDataLoader(train_data, batch_size=train_batch_size, sampler=sampler,
                                            num_workers=4, prefetch_factor=32)
        test_dataloader = CustomDataLoader(test_data, batch_size=test_batch_size, shuffle=True,
                                           num_workers=1, prefetch_factor=32)

        train_accuracy_list = []
        test_accuracy_list = []

        for epoch in range(epochs):
            '''
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
            '''
            test_num = torch.zeros(12)
            test_correct = torch.zeros(12)
            test_complete = 0
            test_total = test_data.__len__()

            with torch.no_grad():
                model.eval()
                print("Test Progress: {:0.5g} %".format(0.), end="")
                for data in test_dataloader:
                    test_x = data[0].to(device)
                    test_y = data[1].to(device)

                    outputs = model(test_x)

                    _, predicted = torch.max(outputs.data, 1)
                    test_complete += test_y.shape[0]

                    answer = torch.argmax(test_y, dim=1)
                    for i in range(test_y.shape[0]):
                        test_num[answer[i]] += 1
                        if predicted[i] == answer[i]:
                            test_correct[answer[i]] += 1

                    print("\rTest Progress: {:0.5g} %".format(100 * test_complete / test_total), end="")

            test_accuracy = 100 * torch.sum(test_correct) / test_total
            test_accuracy_list.append(test_accuracy)
            print("\rTest Accuracy: {0:.3f} %".format(test_accuracy))

            print(test_num)
            test_class_accuracy = 100 * test_correct / test_num
            for i, class_accuracy in enumerate(test_class_accuracy):
                print("class {0:d}: {1:.3f} %".format(i, class_accuracy))



        end = time.time()
        print("총 학습 시간 : {}".format(end - start))
"""
        plt.plot(test_accuracy_list, '-r', label="Test Accuracy")
        plt.plot(train_accuracy_list, '-b', label="Training Accuracy")
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.text(0, min(min(test_accuracy_list), min(train_accuracy_list)) + 1,
                 "isTorch = False\n" +
                 "size = {0}, out_channel = {1}, {2}, lr = {3:f}"
                 .format(size, out_channel1, out_channel2, learning_rate))
        plt.show()
"""