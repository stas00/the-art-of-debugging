import torch

a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
b = torch.tensor([[2., 2., 2.], [3., 3., 3.]], requires_grad=True)

c = a * b
s = c.sum(dim=1)
loss = (s * s).sum()
loss.backward()
print(loss)