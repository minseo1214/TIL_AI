import torch
w=torch.tensor(2.0,requires_grad=True)# 텐서에 대하 기울기 저장
y=w**2
z=2*y+5
z.backward()
print('수식을 w로 미분한 값 : {}'.format(w.grad))
