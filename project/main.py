import time
import torch
import torch.nn as nn
from project.dataloader import image_load

from project.cnn import CNN

if __name__ == '__main__':
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    # device = "cpu"


    start = time.time()

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    load_start = time.time()

    x_train, y_train, x_test, y_test = image_load.load_data()
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    x_test = torch.from_numpy(x_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)


    load_end = time.time()
    print('데이터 로드 시간 : {}'.format(load_end - load_start))

    total_batch = len(x_train)
    print('총 배치의 수 : {}'.format(total_batch))

    epochs = 100
    batch_size = 50


    for epoch in range(epochs):
        avg_cost = 0
        prediction_train = []
        for i in range(0, total_batch, batch_size):
            max_idx = min(i + batch_size, total_batch)
            batch_x = x_train[i:max_idx].to(device)
            batch_y = y_train[i:max_idx].to(device)

            optimizer.zero_grad()
            hypothesis = model(batch_x)
            cost = criterion(hypothesis, batch_y)
            cost.backward()
            optimizer.step()
            avg_cost += cost / total_batch
            prediction_train.append(hypothesis.to(device ='cpu'))

        prediction_train = torch.cat(prediction_train, 0)
        correct_prediction_train = torch.argmax(prediction_train, dim=1) == y_train
        train_accuracy = correct_prediction_train.float().mean()

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
        print('training Accuracy:', train_accuracy.item())
        prediction = model(x_test)
        correct_prediction = torch.argmax(prediction, dim=1) == y_test
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())

    end = time.time()
    print('총 학습 시간 : {}'.format(end - start))



