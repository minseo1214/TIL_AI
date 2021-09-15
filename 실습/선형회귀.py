import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)#seed 설정
x_train=torch.FloatTensor([[1],[2],[3]])#x_train이라는 텐서를 생성합니다
y_train=torch.FloatTensor([[2],[4],[6]])
#x_train의 크기 출력
print(x_train)
print(x_train.shape)
#y_train의 크기 출력
print(y_train)
print(y_train.shape)
#weight와 bias 초기화
W=torch.zeros(1,requires_grad=True)#우선 W을 0으로 초기화
print(W)
b=torch.zeros(1,requires_grad=True)#b을 0으로 초기화
print(b)
hypothesis=x_train*W+b#직선의 방정식에 해당되는 가설 선언
print(hypothesis)
#비용함수 선언하기
cost=torch.mean((hypothesis-y_train)**2)
print(cost)
#경사 하강법구현하기
#SGD는 경사하강법의 일종임
optimizer=optim.SGD([W,b],lr=0.01)#lr은 learning rate
print(optimizer)
optimizer.zero_grad() #미분을 통해 얻은 기울기를 0으로 초기화한다
#기울기를 초기화해야만 새로운 가중치 편향에 대해서 새로운 기울기를 구할 수 있다
# 비용 함수를 미분하여 기울기 계산
cost.backward
# W와 b를 업데이트
optimizer.step() 

nb_epochs = 1999 # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * W + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))
