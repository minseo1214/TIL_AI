#nn.Module로 구현하는 선형회귀
#지금까지는 비용함수와 선형 회귀모델을 직접선언하여 구현했지만 파이토치에서 이미 구현되어있다.
#이미 구현되어져 제공되고 있는 함수들을 불러오는 것으로 더 쉽게 선형 회귀 모델을 구현해보겠다.
#선형 회귀 모델이 nn.Linear()라는 함수로
#평균 제곱오차가 nn.functional.mse_loss()라는 함수로 구현되어져 있다

#단순 선형 회귀 구현하기
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

model = nn.Linear(1,1)#모델을 선언 및 초기화. 단순 선형이므로 input=1,output=1
#model에는 가중치 W와 편향 b가 저장되어져 있다 확인 해보자.
print(list(model.parameters()))
#2개의 값이 출력되는데 첫번째 값이 W이고 두번째 값이 b에 해당된다.
#두 값 모두 현재는 랜덤 초기화되어있다.
#그리고 두 값모두 학습의 대상이므로 requires_grad=True가 되어져 있다.

# optimizer 설정. 경사 하강법 SGD를 사용하고 learning rate를 의미하는 lr은 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward() # backward 연산
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
