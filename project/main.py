import time
import torch
import torch.nn as nn
from project.cnn import CNN
from torchvision.transforms import ToTensor, Lambda
from custom_dataset import CustomDataset
from custom_dataloader import CustomDataLoader

if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    start = time.time()

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    load_start = time.time()

    train_data = CustomDataset(annotations_file='train_data.csv',
                               img_dir='dataset/garbage_classification',
                               target_transform=Lambda(lambda y: torch.zeros(12, dtype=torch.float)
                                                       .scatter_(0, torch.tensor(y), value=1)))

    test_data = CustomDataset(annotations_file='test_data.csv',
                              img_dir='dataset/garbage_classification',
                              target_transform=Lambda(lambda y: torch.zeros(12, dtype=torch.float)
                                                      .scatter_(0, torch.tensor(y), value=1)))

    train_batch_size= 50
    train_dataloader = CustomDataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_dataloader = CustomDataLoader(test_data, batch_size=30, shuffle=False)

    # x_train, y_train, x_test, y_test = garbage.load_data()
    # x_train = torch.from_numpy(x_train).float()
    # y_train = torch.from_numpy(y_train).long()
    # x_test = torch.from_numpy(x_test).float().to(device)
    # y_test = torch.from_numpy(y_test).long().to(device)

    # load_end = time.time()
    # print('데이터 로드 시간 : {}'.format(load_end - load_start))

    # total_batch = len(x_train)
    # print('총 배치의 수 : {}'.format(total_batch))

    epochs = 100
    for epoch in range(epochs):

        avg_cost = 0.0
        for i, data in enumerate(train_dataloader, 0):
            batch_x = data[0].to(device)
            batch_y = data[1].to(device)

            optimizer.zero_grad()

            hypothesis = model(batch_x)
            cost = criterion(hypothesis, batch_y)
            cost.backward()
            optimizer.step()
            avg_cost += cost.item() / len(train_dataloader) * train_batch_size

            if i % train_batch_size == 0:
                print(i)

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

        # prediction = model(x_test)
        # correct_prediction = torch.argmax(prediction, dim=1) == y_test
        # accuracy = correct_prediction.float().mean()
        # print('Accuracy:', accuracy.item())

        correct = 0
        total = 0
        with (torch.no_grad()):
            for data in test_dataloader:
                test_x = data[0].to(device)
                test_y = data[1].to(device)

                outputs = model(test_x)

                _, predicted = torch.max(outputs.data, 1)
                total += test_y.size(0)
                correct += (predicted == torch.argmax(test_y, dim=1)).sum().item()

                # print(predicted)
                # print(torch.argmax(test_y, dim=1))

        print(f'Accuracy of the network on the test images: {100 * correct // total} %')

    end = time.time()
    print('총 학습 시간 : {}'.format(end - start))



