import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import knn

iris = load_iris()

target_names = iris.target_names  # 표본 list

training_data, training_target, test_data, test_target = [], [], [], [] # 각 학습 데이터, 학습 데이터 실측값, 테스트 데이터, 테스트 데이터 실측값

for i in range(0, len(iris.data)):  # 14/15 은 training data로, 1/15는 test data로 분류합니다.
    if (i + 1) % 15 == 0:
        test_data.append(iris.data[i])
        test_target.append(iris.target[i])
    else:
        training_data.append(iris.data[i])
        training_target.append(iris.target[i])


print("Majority Vote") # majority vote 실행합니다.
for i in range(3):
    k = i*2+3 # k = 3,5,7 입니다.
    print("K: ", k)
    majority_vote = knn.majority_vote(k, test_data, training_data, training_target) # KNN의 majority vote 메소드를 실행합니다.
    matching_rate = 0.0 # 결괏값을 0으로 초기화합니다.
    for idx in range(len(test_target)):
        print("Test Data Index: ", idx, "Computed class: ", target_names[majority_vote[idx]], ", True class: ",
              target_names[test_target[idx]])
        if majority_vote[idx] == test_target[idx]: # KNN의 majority vote 메소드 결괏값과 테스트 데이터 실측값의 비교를 통해 일치하면 결괏값에 1을 더합니다.
            matching_rate += 1
    matching_rate /= len(test_target) # 최종 결괏값에 테스트 갯수만큼 나눠 정확도를 계산합니다.
    print("정확도: ",matching_rate)
    print("------------------------------------------------------------------------------")

print("Weighted Majority Vote") # weighted majority vote 실행합니다.
for i in range(3):
    k = i*2+3 # k = 3,5,7 입니다.
    print("K: ", k)
    majority_vote = knn.weighted_majority_vote(k, test_data, training_data, training_target) # KNN의 weighted majority vote 메소드를 실행합니다.
    matching_rate = 0.0 # 결괏값을 0으로 초기화합니다.
    for idx in range(len(test_target)):
        print("Test Data Index: ", idx, "Computed class: ", target_names[majority_vote[idx]], ", True class: ",
              target_names[test_target[idx]])
        if majority_vote[idx] == test_target[idx]: # KNN의 weighted majority vote 메소드 결괏값과 테스트 데이터 실측값의 비교를 통해 일치하면 결괏값에 1을 더합니다.
            matching_rate += 1
    matching_rate /= len(test_target)  # 최종 결괏값에 테스트 갯수만큼 나눠 정확도를 계산합니다.
    print("정확도: ",matching_rate)
    print("------------------------------------------------------------------------------")

