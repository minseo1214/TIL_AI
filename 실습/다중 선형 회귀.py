#다중 선형 회귀(Multivariable Linear regression)
#x의 갯수를 3개로 늘려보자
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
x1_train=torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train=torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train=torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train=torch.FloatTensor([[152], [185], [180], [196], [142]])
#가중치와 편향을 선언할건데, x의 갯수가 3개이기때문에 가중치도 3개로 선언한다.
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)#1e-5는 1*10^-5
nb_epochs =1000
for epoch in range(nb_epochs+1):
  #H(x)계산
  hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3+b
  # cost 계산
  cost = torch.mean((hypothesis - y_train) ** 2)
  optimizer.zero_grad()#0으로 초기화
  cost.backward()#미분
  optimizer.step()#다음 스텝으로 넘어간다.

  if epoch % 100 ==0 :
    print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
        ))
    ### 행렬 연산을 고려하여 파이토치로 구현하기
#Dot Product
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  80], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
print(x_train.shape)
print(y_train.shape)
# 가중치와 편향 선언
W = torch.zeros((3, 1), requires_grad=True)
b= torch.zeros(1,requires_grad=True)
nb_epochs = 20
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    # 편향 b는 브로드 캐스팅되어 각 샘플에 더해집니다.
    hypothesis = x_train.matmul(W) + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))
