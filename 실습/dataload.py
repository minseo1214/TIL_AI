#파이토치에서 데이터를 좀 더 쉽게 다를 수 있도록 유용한 도구로서 데이터셋(Dataset)과 데이터로더(DataLoader)를 제공합니다.
#이를 사용하면 미니 배치 학습, 데이터 셔플(shuffle),병렬처리까지 간단히 수행할 수 있습니다.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset#텐서 데이터 셋
from torch.utils.data import DataLoader # 데이터 로더
#TensorData은 기본적으로 텐서를 입력으로 받습니다.
x_train= torch.FloatTensor([[73, 80, 75],
                            [93, 88, 93], 
                            [89, 91, 90], 
                            [96, 98, 100],   
                            [73, 66, 70]])
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
dataset = TensorDataset(x_train,y_train) #TensorDataset의 입력으로 사용하고 dataset을 저장합니다.
#파이토치의 데이터셋을 만들었다면 데이터 로더를 사용가능합니다.
#데이터 로더는 기본적으로 2개의 인자를 입력받습니다.
#하나는 데이터셋, 미니배치의 크기입니다.
#추가적으로 사용되는 인자로는 shuffle이 있는데 shuffle=True를 선택하면 Epoch마다 데이셋을 섞어서 데이터가 학습되는 순서를 바꿉니다.
#계속 같은 데이터셋으로 학습시 overfitting문제가 있을 수 있음.
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#옵티마이저(최적화 알고리즘)설계
model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 
nb_epochs = 20
for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    # print(batch_idx)
    # print(samples)
    x_train, y_train = samples
    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(epoch, nb_epochs, batch_idx+1, len(dataloader),cost.item()))
    # 임의의 입력 [73, 80, 75]를 선언
new_var =  torch.FloatTensor([[73, 80, 75]]) 
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 
