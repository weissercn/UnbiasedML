import torch
import models
import utils


print("hi")

# class LegendreIntegral(Function):w

from torch.autograd import gradcheck

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.

test_class = utils.LegendreIntegral

input = (torch.randn(20,dtype=torch.double,requires_grad=True))
test = gradcheck(test_class, input, eps=1e-6, atol=1e-4)

print(test)

