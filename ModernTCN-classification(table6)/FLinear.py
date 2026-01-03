import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch import Tensor
from scipy.special import gamma 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def clip_matrix_norm(matrix, max_norm):
    norm = torch.norm(matrix)
    if norm > max_norm:
        matrix = matrix * (max_norm / norm)
    return matrix

class Fractional_Order_Matrix_Differential_Solver(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input1,w,b,alpha,c):
        alpha = torch.tensor(alpha)
        c = torch.tensor(c)
        ctx.save_for_backward(input1,w,b,alpha,c)
        outputs = input1@w + b
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        input1,w,b,alpha,c = ctx.saved_tensors
        x_fractional, w_fractional = Fractional_Order_Matrix_Differential_Solver.Fractional_Order_Matrix_Differential_Linear(input1,w,b,alpha,c)   
        x_grad = grad_outputs@x_fractional
        w_grad = w_fractional@grad_outputs
        b_grad = grad_outputs.sum(dim=0)
        return x_grad, w_grad, b_grad,None,None
          
    @staticmethod
    def Fractional_Order_Matrix_Differential_Linear(xs,ws,b,alpha,c):
        wf = ws[:,0].view(1,-1)
        #main
        w_main = torch.mul(xs,(torch.abs(wf)+1e-8)**(1-alpha)/gamma(2-alpha))
        #partial
        w_partial = torch.mul((xs@wf.T).expand(xs.shape) - torch.mul(xs,wf) + b[0], torch.sgn(wf)*(torch.abs(wf)+1e-8)**(-alpha)/gamma(1-alpha))
        return ws.T, (w_main + clip_matrix_norm(w_partial,c)).transpose(-2,-1)

class FLinear(nn.Module):
    
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, alpha=0.9, c=1.0, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.c = c
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return Fractional_Order_Matrix_Differential_Solver.apply(x, self.weight.T, self.bias, self.alpha,self.c)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"