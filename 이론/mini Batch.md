<h2>미니 배치와 배치크기(Mini batch and Batch Size)</h2><br>
x_train = torch.FloatTensor([[73, 80, 75],<br>
현재는 데이터가 적지만 방대한 데이터에 대해 경사하강법을 수행하는 것은 매우 느릴 뿐만 아니라 많은 계산량 필요<br>
메모리의 한계로 계산이 불가능 한 경우도 있다.<br>
때문에 전체 데이터를 더 작은 단위로 나누어서 해단 단위로 학습하는 개념이 나오게 되었다.<br>
이때 이 단위를 미니배치(Mini Batch)라고 한다.<br>

이 그림처럼 나누어 미니배치를 학습하게되면 미니배치에 대한 비용(cost)를 계산하고 경사하강법을 수행합니다.<br>
<img alt="image" src="https://wikidocs.net/images/page/55580/%EB%AF%B8%EB%8B%88%EB%B0%B0%EC%B9%98.PNG"><br>
그리고 마지막 배치까지 이를 반복하는데 이렇게 전체 데이터에 대한 학습이 1회 끝나면 1에포크(Epoch)라고 합니다.<br>
따라서 미니배치의 갯수에 따라 1epoch는 달라집니다.<br>
minibatch의 크기를 batchsize라고합니다.<br>
전체 데이터에 대해 한번에 경사하강법을 수행하는 방법을 '배치 경사 하강법'이라고하고, 미니 배치단위로 경사하강법을 수행하는 것을 '미니배치 경사하강법'이라고 합니다.<br>
배치 경사하강법은 경사 하강법을 할 때 전체 데이터를 사용하기 때문에 가중치 값이 최적값에 수렵하는 과정이 매우 안정적이지만, 계산량이 너무 많이 듭니다.<br>
미니 배치 경사하강법은 경사 하강법을 할 때, 전체 데이터의 일부만을 보고 수행하므로 죄적값으로 수렴하는 과정에서 값이 조금 헤매기도 하지만 훈련 속도가 빠릅니다.<br>
배치크기는 보토 2의 제곱수를 사용합니다.ex)2^1=2,2^2=4,2^3=8... 그 이유는 CPU와 GPU의 메모리가 2의 배수이므로 배치크기가 2의 제곱수일경우 데이터 송수신의 효율을 높일 수 있습니다.<br>

<br><br><br><br>

<h2>이터레이션(Iteration)</h2>
<img alt="image" src="https://wikidocs.net/images/page/36033/batchandepochiteration.PNG"><br>
Iteration은 한번의 epoch내에서 이루어지는 매개변수인 가중치W와 b의 업데이트 횟수입니다<br>
전체 데이터가 2,000일 때 배치 크기를 200으로 한다면 이터레이션의 수는 총 10개 입니다.<br>
이는 한번에 epoch당 매개변수 업데이트가 10번 이루어짐을 의미합니다.<br>
