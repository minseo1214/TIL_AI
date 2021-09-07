
### U-Net
**Biomedical 분야에서 Image Segmentation을 목적으로 제안된 End-to-End방식의  Fully-Convolutional network 기반 모델**

#### U-Net으로 들어가기 전에
> End-to-End
입력에서 출력까지 파이프라인 네트워크 없이 한 번에 처리한다는 뜻이다.
(end-to-end deep learning 을 사용하기 위해서는, 파이프라인 네트워크 없이도 잘 동작 할 만큼의 많은 데이터가 필요하다.
문제가 복잡할 수록, 전체 문제(전체 네트워크)를 간단한 문제(파이프라인 네트워크)로 나눠서 해결하는 것이 효율적일 때도 있기 때문이다.)

### U-Net?

![](https://images.velog.io/images/minseo1214/post/ad55fcff-8b21-4790-84dc-0126a1e5d7de/image.png)

![](https://images.velog.io/images/minseo1214/post/23078173-7c76-4632-a718-05b101ce037b/image.png)
왼쪽은 Contracting path(Encoding),오른쪽(Decoding)로 정의 하였다.

> 1. Convolution Encoder에 해당하는 Contracting Path + Convolution Decoder에 해당하는 Expanding Path의 구조로 구성. (해당 구조는 Fully Convolution + Deconvolution 구조의 조합)

> 2.Expanding Path에서 Upsampling 할 때, 좀 더 정확한 Localization을 하기 위해서 Contracting Path의 Feature를 Copy and Crop하여 Concat(결합) 하는 구조.

> Data Augmentation(증폭)


 U-Net의 main idea는 다음과 같다.

 1) Contracting Path : 이미지의 context 포착(맥락)
 2) Expansive Path : feature map을 upsampling 하고 1)에서 포착한 feature map의  context와 결합 → 이는 더욱 정확한 localization을 하는 역할을 한다.
 
 기존의 FCN과 다른 중요한 점 2가지가 있다.
 1) upsampling 과정에서 feature chennel 수가 많다. 이는 context를 resolution successive layer에게 전파할 수 있음을 의미한다.
  2) 각 Convolution의 valid part만 사용한다.(valid part란, full context가 들어있는 segmentation map을 의미) 이는 Overlap-tile 기법을 사용하여 원활한 segmentation이 가능하도록 한다.

![](https://images.velog.io/images/minseo1214/post/be5974dc-180d-4d8e-8590-9891cfb8b97e/image.png)
그림에서 보이듯이 Blue tile과 Yellow tile이 서로 Overlap되어 있음을 확인할 수 있으며, Blue area에 기반하여 Yellow area의 Segmentation을 prediction(예측) 했음을 확인 할 수 있다. 그 이후, missing data는 mirroring을 통하여 extrapolation한다.
기존에는 Sliding-window을 하면서 로컬 영역(패치)을 입력으로 제공해서 각 픽셀의 클래스 레이블을 예측했지만, 이 방법은 2가지 단점으로 인해서 Fully Convolution Network구조를 제안하고 있다.
>네트워크가 각 패치에 대해 개별적으로 실행되어야 하고 패치가 겹쳐 중복성이 많기 때문에 상당히 느리다.
localization과 context사이에는 trade-off가 있는데, 이는 큰 사이즈의 patches는 많은 max-pooling을 해야해서 localization의 정확도가 떨어질 수 있고, 반면 작은 사이즈의 patches는 협소한 context만을 볼 수 있기 때문
Contracting Path에서 Pooling되기 전의 Feture들은 Upsampling 시에 Layer와 결합되어 고 해상도 output을 만들어 낼 수 있다.

하나 더 중요한 점은! 많은 수의 Feature Channels를 사용하는데 아래 네트워크 아키텍쳐를 보시면 DownSampling시에는 64 채널 -> 1024채널까지 증가 되고, UpSampling시에는 1024 채널 -> 64채널을 사용하고 있다.

네트워크는 fully connected layers를 전혀 사용하지 않고, 각 layer에서 convolution만 사용한다.

다음으로, U-Net에서는 Segmentation시 overlab-tile 전략을 사용한다.

![](https://images.velog.io/images/minseo1214/post/2ca78806-a4af-4970-8d49-2f54bf5baa76/image.png)

> Overlap-tile 전략은, U-Net에서 다루는 전자 현미경 데이터의 특성상 이미지 사이즈의 크기가 상당히 크기 때문에 Patch 단위로 잘라서 Input 으로 넣고 있다.

> 이때 Fig.2에서 보는 것과 같이 Border 부분에 정보가 없는 빈 부분을 0으로 채우거나, 주변의 값들로 채우거나 이런 방법이 아닌 Mirroring 방법으로 pixel의 값을 채워주는 방법이다.

> 노랑색 영역이 실제 세그멘테이션 될 영역이고, 파랑색 부분이 Patch이다.

> 그림을 확대해서 자세히 보시면, 거울처럼 반사되어 border부분이 채워진 것을 확인 할 수 있었다.

![](https://images.velog.io/images/minseo1214/post/a31fe39b-2866-4cf4-8e53-b19c7d407dfa/image.png)

Overlap-tile 이라는 이름은, 파랑색 부분이 Patch단위로 잘라서 세그멘테이션을 하게 되는데 (용어는, Patch == Tile) 이 부분이 아래 그림처럼 겹쳐서 뜯어내서 학습시키기 때문인 것 같다

(2) Architecture

 이제, U-Net의 구조를 자세하게 살펴보자. 빠른 이해를 위해 위의 그림을 다시 가지고 오면 다음과 같으며, Contracting Path와 Expansive Path 두 가지로 나누어보겠다.
 ![](https://images.velog.io/images/minseo1214/post/e5940d7d-3714-4906-8e60-fd73513507e1/image.png)
 
  1) Contracting Path

 Contracting Path는 일반적인 CNN을 따르며, Downsampling을 위한 Stride2, 2x2 max pooling 연산과 ReLU를 포함한 두 번의 반복된 3x3 unpadded convolutions 연산을 거친다. 

 즉, 3x3conv → ReLU → 2x2 max pooling → 3x3conv → ReLU → 2x2 max pooling 이다.

 그리고 downsampling 과정에서 feature map channel의 수는 2배로 증가시킨다.

 

 2) Expansive Path

 Expansive Path는 2x2 convolutions를 통해 upsampled된 feature map과 1)의 cropped feature map과의 concatenation 후에 ReLU 연산을 포함한 두 번의 3x3 convolutions 연산을 거친다.
 
 이 때, 1)의 Crop된 featrue map을 보내는 이유는 Convolution을 하면서 border pixel에 대한 정보가 날아가는 것에 대한 양쪽의 보정을 위함이다.

 마지막 layer는 1x1 convolution 연산을 하는데 이는 64개의 component feature vector를 desired number of classes에 mapping 하기 위함이다.

3. Training
학습은 Stochastic gradient descent 로 구현되었다.

이 논문에서는 학습시에 GPU memory의 사용량을 최대화 시키기 위해서 batch size를 크게해서 학습시키는 것 보다 input tile 의 size를 크게 주는 방법을 사용하는데, 이 방법으로 Batch Size가 작기 때문에, 이를 보완하고자 momentum의 값을 0.99값을 줘서 과거의 값들을 더 많이 반영하게 하여 학습이 더 잘 되도록 하였다.
![](https://images.velog.io/images/minseo1214/post/22ab4302-9f0c-4123-9f37-bdac6dc02444/image.png)
