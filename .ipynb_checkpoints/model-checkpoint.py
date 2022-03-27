#from statistics import mode
#from zmq import device
import math
from models.gdn import GDN
from models import prob_culmulative, models
import torch
from torch import nn 
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

class Network(nn.Module):
    def __init__(self, device, N=128, M=192, training=True):
        super().__init__()

        self.device = device
        self.training = training
        # 用来计算hyperprior z的熵
        self.entropy_bottleneck = EntropyBottleneck(N)
        # 利用hyperprior得出的方差，优化隐式表示y的熵模型
        self.gaussian_conditional = GaussianConditional([.11, 256.]) 

        self.g_analysis = nn.Sequential(
            nn.Conv2d(3, N, 5, stride=2, padding=2),
            GDN(N, device=self.device),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            GDN(N, device=self.device),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            GDN(N, device=self.device),
            nn.Conv2d(N, M, 5, stride=2, padding=2),
        )

        self.h_analysis = nn.Sequential(
            nn.Conv2d(M, N, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(N, N, 5, stride=2, padding=2)
        )

        self.g_synthesis = nn.Sequential(
            nn.ConvTranspose2d(M, N, 5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True, device=self.device),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True, device=self.device),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True, device=self.device),
            nn.ConvTranspose2d(N, 3, 5, stride=2, padding=2, output_padding=1)
        )

        self.h_synthesis = nn.Sequential(
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(N, M, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.g_analysis(x)
        
        # print('hhh')
        # print(y.shape)
        z = self.h_analysis(y)
        # print(z.shape)
        z_hat, z_likelihodds = self.entropy_bottleneck(z)
        # print(z_hat.shape)
        scale_hat = self.h_synthesis(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scale_hat)

        x_hat = self.g_synthesis(y_hat)

#         if self.training:
#             noise = nn.init.uniform_(torch.empty(y.shape), -0.5, 0.5)  
#             noise = noise.to(self.device)
#             #print(self.device)
#             #print(noise.shape)
#             q = y + noise
#             #print(q)
#         else:
#             q = torch.round(y)
        #x_hat = self.decode(q)


    
        #clipped_X_hat = X_hat.clamp(0., 1.)

#         def calculate_rate(Y):
#             cumulative = prob_culmulative.Culmulative(Y.shape[1]).to(self.device)
#             # 两个概率累积的差值即为对应点的概率
#             p_y = cumulative(Y + 0.5) - cumulative(Y - 0.5)
#             sum_of_bits = torch.sum(-torch.log2(p_y))
#             # 这里不确定要不要除以输入图片的通道数
#             return sum_of_bits / (x.shape[0] * x.shape[2] * x.shape[3])
        
        def calculate_rate(y_likelihoods):
            N, _, H, W = x.size()
            num_pixels = N * H * W
            return torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)
            
        #rate = calculate_rate(q)
        rate = calculate_rate(y_likelihoods) + calculate_rate(z_likelihodds)
        #distortion = torch.mean(torch.square(X - X_hat))
        return x_hat, rate



class Compress_and_DeCompress(nn.Module):
    def __init__(self, device, training=False):
        super().__init__()
        self.analysis = models.Analysis(device=device)
        # self.density = density.Density()
        self.synthesis = models.Synthesis(device=device)
        self.training = training
        self.device = device

    def forward(self, X):
        Y = self.analysis(X)
        if self.training:
            noise = nn.init.uniform_(torch.empty(Y.shape), -0.5, 0.5)  
            noise = noise.to(self.device)
            #print(self.device)
            #print(noise.shape)
            q = Y + noise
            #print(q)
            
        else:
            q = torch.round(Y)

        X_hat = self.synthesis(q)
        #clipped_X_hat = X_hat.clamp(0., 1.)

        def calculate_rate(Y):
            cumulative = prob_culmulative.Culmulative(Y.shape[1]).to(self.device)
            # 两个概率累积的差值即为对应点的概率
            p_y = cumulative(Y + 0.5) - cumulative(Y - 0.5)
            sum_of_bits = torch.sum(-torch.log2(p_y))
            # 这里不确定要不要除以输入图片的通道数
            return sum_of_bits / (X.shape[0] * X.shape[2] * X.shape[3])

        rate = calculate_rate(q)
        #distortion = torch.mean(torch.square(X - X_hat))

        return X_hat, rate
        



