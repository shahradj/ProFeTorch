import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Trend', 'FourierModel', 'Seasonal', 'Squasher', 'Holiday', 'HolidayRange']

class Trend(nn.Module):
    """
    Broken Trend model, with breakpoints as defined by user.
    """
    def __init__(self, breakpoints=None):
        super().__init__()
        self.bpoints = breakpoints
        self.init_layer = nn.Linear(1,1) # first linear bit
        self.init_layer.weight.data = torch.Tensor([[0.]])
        self.init_layer.bias.data = torch.Tensor([[0.]])
            
        if breakpoints:
            # create deltas which is how the gradient will change
            deltas = torch.zeros(len(breakpoints)) # initialisation
            self.deltas = nn.Parameter(deltas) # make it a parameter
        
    def __copy2array(self):
        """
        Saves parameters into wb
        """
        # extract gradient and bias
        w = self.init_layer.weight
        b = self.init_layer.bias
        self.params = [[w,b]] # save it to buffer
        if self.bpoints:
            for d, x1 in zip(self.deltas, self.bpoints):
                y1 = w *x1 + b # find the endpoint of line segment (x1, y1)
                w = w + d # add on the delta to gradient 
                b = y1 - w * x1 # find new bias of line segment 
                self.params.append([w,b]) # add to buffer

        # create buffer
        self.wb = torch.zeros(len(self.params), len(self.params[0]))
        for i in range(self.wb.shape[0]):
            for j in range(self.wb.shape[1]):
                self.wb[i,j] = self.params[i][j]
        
    def forward(self, x):
        if self.bpoints:
            self.__copy2array() # copy across parameters into matrix
            # get the line segment area (x_sec) for each x
            x_sec = x >= self.bpoints
            x_sec = x_sec.sum(1)
            
            # get final prediction y = mx +b for relevant section
            return x*self.wb[x_sec][:,:1] + self.wb[x_sec][:,1:]
        
        else:
            return self.init_layer(x)


class FourierModel(nn.Module):
    def __init__(self, p=365.25, n=7, init=None):
        super().__init__()
        self.p, self.n = p, n
        np = [(i+1, p) for i in range(n)]
        self.np = np
        if n > 0:
            self.linear = nn.Linear(n, 1, bias=False)
            # initialise weight parameters
            if init:
                self.linear.weight.data = torch.Tensor(init)
            else:
                self.linear.weight.data = torch.Tensor(torch.zeros_like(self.linear.weight.data))
        
    def forward(self, x):
        if self.n > 0:
            cos = [torch.cos(2*np.pi*n*x/p) for n,p in self.np]
            sin = [torch.sin(2*np.pi*n*x/p) for n,p in self.np]

            return self.linear(torch.cat(cos + sin, dim=1))
        
        else:
            return 0
    
    def plot(self):
        if self.n > 0:
            x = torch.linspace(0, self.p, steps=100)
            y = self.forward(x[:,None])
            plt.figure(figsize=(12,5))
            plt.plot(x.cpu().numpy(), y.detach().cpu().numpy())
            plt.show()


class Seasonal(nn.Module):
    def __init__(self, y_n=7, m_n=5, w_n=0, 
                 y_p=365.25, m_p=30.5, w_p=7):
        super().__init__()
        # Calculate Xavier Glorot initialisation
        fourier_components = 2 * np.array([y_n, m_n, w_n])
        idxs = np.cumsum(fourier_components)
        fan_in = idxs[-1] + 1 # +1 due to trend
        fan_out = 1
        std = np.sqrt(2/(fan_in + fan_out))
        w = torch.randn(1, idxs[-1]) * std
        w = torch.clamp(w, -2*std, 2*std)
        
        self.yearly = FourierModel(y_p, y_n, w[:,:idxs[0]])
        self.monthly = FourierModel(m_p, m_n, w[:,idxs[0]:idxs[1]])
        self.weekly = FourierModel(w_p, w_n, w[:,idxs[1]:idxs[2]])
        
    def forward(self, x):
        return self.yearly(x) + self.monthly(x) + self.weekly(x)
    
class DefaultModel(nn.Module):
    def __init__(self, breakpoints=None, y_n=7, m_n=5, w_n=0):
        super().__init__()
        self.trend = Trend(breakpoints)
        self.seasonal = FourierModel(y_n, m_n, w_n)

    def forward(self, x):
        return self.seasonal(x) +self.trend(x)
        


class Squasher(nn.Module):
    def __init__(self, low, high, alpha=0.01):
        super().__init__()
        self.L, self.H, self.alpha = low, high, alpha
    def forward(self, x): 
        x[x < self.L] = self.alpha * (x[x < self.L] - self.L) + self.L
        x[x > self.H] = self.alpha * (x[x > self.H] - self.H) + self.H
        return x


class Holiday(nn.Module):
    def __init__(self, holiday, repeat_every=365):
        super().__init__()
        self.holiday = holiday
        self.repeat_every = repeat_every
        self.w = nn.Parameter(torch.zeros(1)+0.05)
        
    def forward(self, t):
        rem = torch.remainder(t - self.holiday, self.repeat_every)
        return (rem == 0).float() * self.w


class HolidayRange(nn.Module):
    def __init__(self, holidays):
        """
        holidays: list of lists containing lower and upper bound of hols
        """
        super().__init__()
        self.holidays = holidays
        self.w = nn.Parameter(torch.zeros(1)+0.05)
        
    def forward(self, t):
        bounded = [(l<=t) & (t<=u) for l,u in self.holidays]
        return sum(bounded).float()*self.w
