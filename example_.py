import torch

cte1 = torch.rand(5, 3).requires_grad_(False)
cte2 = torch.rand(5, 5).requires_grad_(False)
tensor = torch.ones(3, 5).requires_grad_()

optim = torch.optim.SGD([tensor], lr=1)

print(f'Initial tensor'
      f'{tensor}')
for i in range(5):
    optim.zero_grad()
    print(f'Iteration {i}')
    output = cte1 @ tensor + cte2
    print(f'Requires grad? {output.requires_grad}')
    output.sum().backward()
    print(f'Tensor gradients \n'
              f' {tensor.grad}')
    optim.step()
    print(f'Tensors \n'
              f' {tensor}')
