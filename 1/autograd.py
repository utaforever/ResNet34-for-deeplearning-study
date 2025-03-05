
import matplotlib.pyplot as plt
import numpy as np
import torch

# 创建x张量，并设置requires_grad=True以跟踪对其的操作  
x = torch.linspace(-3 * np.pi, 3 * np.pi, 100, requires_grad=True)

# 计算sin(x)  
y = torch.sin(x)

# 为了得到一个标量，我们可以对y求和（或取其他任何标量函数）  
# 注意：这里求和是为了得到一个标量，以便能够进行反向传播  
# 但这样做意味着我们实际上是在计算整个区间上sin(x)的平均梯度的近似  
y_sum = y.sum()

# 对标量进行反向传播  
y_sum.backward()

# 此时，x.grad将包含sin(x)对x的梯度的近似（基于整个张量的操作）  
# 但由于我们是对整个张量求和，所以每个点的梯度将是相同的（实际上应该接近0，因为sin(x)在完整周期内积分为0）  
# 然而，由于数值稳定性和浮点精度问题，梯度可能不会完全为0  

# 为了可视化，我们可以绘制x、sin(x)和x.grad（尽管x.grad对于这个问题来说不是很有意义）  
# 但为了练习目的，我们可以这样做来观察PyTorch是如何工作的  

# 由于x.grad包含了梯度信息，我们可以将其转换为numpy数组进行绘图  
grad_np = x.grad.detach().numpy()

# 绘制sin(x)和其“梯度”（注意这里的梯度是基于对整个张量操作的近似）  
plt.figure()
plt.plot(x.detach().numpy(), y.detach().numpy(), label='sin(x)')
plt.plot(x.detach().numpy(), grad_np, label='"Gradient" of sin(x) (approximation based on sum)')
plt.legend()
plt.xlabel('x')
plt.ylabel('Value / Gradient')
plt.title('sin(x) and its approximate gradient based on PyTorch autograd')
plt.show()

# 重要的是要理解，这里的“梯度”图并不是真正的sin(x)在每个点上的梯度（即cos(x)）  
# 而是基于我们对整个张量进行标量操作（求和）后得到的梯度的近似  
# 在实践中，我们通常不会这样计算梯度，而是会对具体的标量损失函数进行反向传播