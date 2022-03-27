import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as f
# def gdn(X, gamma, beta):
#     if torch.is_grad_enabled():
#         beta = beta ** 2 - 2 ** (-10)
#         gamma = gamma ** 2 - 2 ** (-10)

#     Y = X / torch.sqrt(beta**2 + 
#         torch.sum(gamma * torch.square(X), dim=0))

#     return Y

# class GDN(nn.Module):
#     def __init__(self, num_features):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.full((1, num_features), 0.1))
#         self.beta = nn.Parameter(torch.full((1, num_features), 1e-6))

#     def forward(self, X):
#         Y = gdn(X, self.gamma, self.beta)
#         return Y


# def igdn(X, gamma, beta):
#     if torch.is_grad_enabled():
#         beta = beta ** 2 - 2 ** (-10)
#         gamma = gamma ** 2 - 2 ** (-10)

#     Y = X * torch.sqrt(beta + torch.sum(gamma + torch.square(X), dim=0))

#     return Y

# class IGDN(nn.Module):
#     def __init__(self, num_features):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.full((1, num_features), 0.1))
#         self.beta = nn.Parameter(torch.full((1, num_features), 1e-6))

#     def forward(self, X):
#         Y = igdn(X, self.gamma, self.beta)
#         return Y


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size(), device=inputs.device)*bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
  
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    """
  
    def __init__(self,
                 ch,
                 device,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.tensor([reparam_offset], device=device)

        self.build(ch, torch.device(device))
  
    def build(self, ch, device):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**.5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch, device=device)+self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch, device=device)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size() 
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()
        #print(inputs.size())
        
        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal 

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        #print(ch)
        gamma  = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
 
        #norm_ = torch.ones(inputs.shape)
        norm_ = f.conv2d(inputs**2, gamma, beta)
        #norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)
  
        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs