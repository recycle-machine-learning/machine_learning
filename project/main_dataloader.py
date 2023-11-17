import time

import torch
import torch.nn as nn
from project.cnn import CNN
from torchvision.transforms import Lambda
import matplotlib.pyplot as plt

from project.dataloader.custom_dataset import CustomDataset
from project.dataloader.custom_dataloader import CustomDataLoader
from project.dataloader.make_csv import save_csv  

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    start = time.time()

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

    load_start = time.time()
    save_csv()

    train_data = CustomDataset(annotations_file="train_data.csv",
                               img_dir="dataset/garbage_classification",
                               target_transform=Lambda(lambda y: torch.zeros(12, dtype=torch.float)
                                                       .scatter_(0, torch.tensor(y), value=1)))

    test_data = CustomDataset(annotations_file="test_data.csv",
                              img_dir="dataset/garbage_classification",
                              target_transform=Lambda(lambda y: torch.zeros(12, dtype=torch.float)
                                                      .scatter_(0, torch.tensor(y), value=1)))

    train_batch_size= 50
    train_dataloader = CustomDataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_dataloader = CustomDataLoader(test_data, batch_size=30, shuffle=False)

    epochs = 10

    test_accuracy_list = []
    train_accuracy_list = []

    for epoch in range(epochs):

        avg_cost = 0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(train_dataloader, 0):
            batch_x = data[0].to(device)
            batch_y = data[1].to(device)

            optimizer.zero_grad()

            hypothesis = model(batch_x)
            cost = criterion(hypothesis, batch_y)
            cost.backward()
            optimizer.step()
            avg_cost += cost.item()

            _, predicted = torch.max(hypothesis.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == torch.argmax(batch_y, dim=1)).sum().item()

            if i % train_batch_size == 0:
                print(i)

        accuracy = 100 * train_correct / train_total
        train_accuracy_list.append(accuracy)
        print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, avg_cost / train_total))
        print("Train Accuracy: {0:.3f} %".format(accuracy))

        test_correct = 0
        test_total = 0
        with (torch.no_grad()):
            for data in test_dataloader:
                test_x = data[0].to(device)
                test_y = data[1].to(device)

                outputs = model(test_x)

                _, predicted = torch.max(outputs.data, 1)
                test_total += test_y.size(0)
                test_correct += (predicted == torch.argmax(test_y, dim=1)).sum().item()

        test_accuracy = 100 * test_correct / test_total
        test_accuracy_list.append(test_accuracy)
        print("Test Accuracy: {0:.3f} %".format(100 * test_correct / test_total))

    end = time.time()
    print("총 학습 시간 : {}".format(end - start))

    plt.plot(test_accuracy_list, '-r', label="Test Accuracy")
    plt.plot(train_accuracy_list, '-b', label="Training Accuracy")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
