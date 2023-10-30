import math
from collections import Counter

import numpy


def majority_vote(k, test_data, training_data, training_target): # majority vote 결과 제공 메소드
    return vote(False, k, test_data, training_data, training_target)


def weighted_majority_vote(k, test_data, training_data, training_target): # weighted majority vote 제공 메소드
    return vote(True, k, test_data, training_data, training_target)

# vote를 진행할 메소드
# is_weighted : weighted majority vote를 진행할지 여부를 받는 파라미터
# K : 고려할 이웃 수
# test_data : 테스트를 진행할 데이터
# training_data : 학습 데이터
# training_target : 학습 데이터 결괏값
def vote(is_weighted, k, test_data, training_data, training_target):
    distances = [] # 각 테스트 데이터에 대한 가장 가까운 k 데이터 중 빈도수가 가장 큰 데이터를 담는 배열
    for test in test_data:
        result = obtain_knn(k, test, training_data) # 테스트 데이터와 학습 데이터 140개를 이용해 가장 가까운 이웃 k개를 가져오는 메소드를 호출합니다.
        result_target = dict() # 각 class에 대해 빈도수를 저장하기 위한 파이썬 dictionary 변수를 생성합니다.
        for target in result:
            value = result_target.get(training_target[target[1]], 0) # k개 데이터 배열에서 각 데이터의 학습 데이터 인덱스를 통해 해당 학습 데이터의 클래스를 순서를 가져옵니다. 없으면 기본값 0을 반환합니다.
            add_value = 1 / target[0] if is_weighted else 1 # weighted majority vote 여부(is_weighted)이 활성화되면 앞서 계산된 거리를 역수를 취해 가중치를 부여합니다. 비활성화면 1의 값을 부여합니다.
            result_target[training_target[target[1]]] = value + add_value # 기존의 값과 거리를 통해 계산한 빈도수를 더 합니다.
        sort_result_target = sorted(result_target.items(), key=lambda x: x[1], reverse=True) # 각 class에 대해서 계산된 빈도수를 통해 내림차순으로 정렬합니다.
        print(sort_result_target)
        nearest_neighbor = sort_result_target[0][0] # 정렬된 값중 가장 큰 값. 즉, 첫번쨰 인덱스에 존재하는 class를 가져옵니다.
        distances.append(nearest_neighbor) # 선택된 class를 배열에 저장합니다.
    return distances


def obtain_knn(k, test, training_data): # 테스트 데이터와 학습 데이터 140개를 이용해 가장 가까운 이웃 k개를 추출하는 메소드
    distance = calculate_distance(test, training_data) # 테스트 데이터와 학습 데이터 140개를 이용해 각 학습 데이터당 거리를 계산 후 정렬한 배열을 반환하는 메소드를 호출합니다.
    result = distance[0:k] # 가장 가까운 k개의 데이터를 가져옵니다.
    return result


def calculate_distance(test, training_data): # 테스트 데이터와 학습 데이터를 이용해 각 학습데이터와 테스트 데이터간의 거리를 계산하는 메소드
    distances = [] # 계산한 거리와 학습 데이터의 인덱스를 기록합니다.
    for idx, train in enumerate(training_data):
        distances.append([math.sqrt(sum(pow(test - train, 2))), idx]) # 거리 계산(Euclidean) 후 [거리,인덱스] 형태로 배열에 저장합니다.
    distances.sort(key=lambda data: data[0])  # 거리 계산 값을 기준으로 오름차순으로 정렬합니다.
    return distances
