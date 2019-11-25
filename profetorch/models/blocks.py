import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import convert_date

__all__ = ['DefaultModel', 'Trend', 'FourierModel', 'Seasonal', 'Squasher', 'Holiday', 'HolidayRange']

class Trend(nn.Module):
    """
    Broken Trend model, with breakpoints as defined by user.
    """
    def __init__(self, breakpoints:int=None, moment=None):
        super().__init__()
        self.init_layer = nn.Linear(1,1) # first linear bit
            
        if breakpoints is not None:
            if isinstance(breakpoints, int):
                # range = moment['t_range'][1] - moment['t_range'][0]
                # breakpoints = torch.rand(breakpoint)*range + moment['t_range'][0]
                if breakpoints > 0:
                    breakpoints = np.linspace(*moment['t_range'], breakpoints+1, endpoint=False)[1:]
                else:
                    breakpoints = None
            else:
                breakpoints = convert_date(breakpoints)
            # create deltas which is how the gradient will change
            deltas = torch.zeros(len(breakpoints)) # initialisation
            self.deltas = nn.Parameter(deltas) # make it a parameter
        
        self.bpoints = breakpoints
        
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
        
    def forward(self, t:torch.Tensor):
        if self.bpoints is not None:
            self.__copy2array() # copy across parameters into matrix
            # get the line segment area (x_sec) for each x
            x_sec = t >= self.bpoints
            x_sec = x_sec.sum(1)
            
            # get final prediction y = mx +b for relevant section
            return t*self.wb[x_sec][:,:1] + self.wb[x_sec][:,1:]
        
        else:
            return self.init_layer(t)

TWOPI = 2*np.pi

class FourierModel(nn.Module):
    def __init__(self, p:float=365.25, scale:float=1, n:int=7):
        super().__init__()
        self.np = [(i+1, p/scale) for i in range(n)]
        if n > 0:
            self.linear = nn.Linear(n * 2, 1, bias=False)
            
    def forward(self, t:torch.Tensor):
        if len(self.np) > 0:
            cos = [torch.cos(TWOPI * n * t / p) for n,p in self.np]
            sin = [torch.sin(TWOPI * n * t / p) for n,p in self.np]

            return self.linear(torch.cat(cos + sin, dim=1))
        
        else:
            return 0
    
    def plot(self):
        if self.n > 0:
            t = torch.linspace(0, self.p, steps=100)
            y = self.forward(t[:,None])
            plt.figure(figsize=(12,5))
            plt.plot(t.cpu().numpy(), y.detach().cpu().numpy())
            plt.show()


class Seasonal(nn.Module):
    def __init__(self, y_n=7, m_n=5, w_n=0, 
                 y_p=365.25, m_p=30.5, w_p=7, scale=1):
        super().__init__()
        self.yearly = FourierModel(y_p, scale, y_n) # , w[:,:idxs[0]]
        self.monthly = FourierModel(m_p, scale, m_n) # w[:,idxs[0]:idxs[1]]
        self.weekly = FourierModel(w_p, scale, w_n) # w[:,idxs[1]:idxs[2]]
        
    def forward(self, t):
        return self.yearly(t) + self.monthly(t) + self.weekly(t)
        

class Squasher(nn.Module):
    def __init__(self, low=None, high=None, mean=0, sd=1, alpha=0.01):
        super().__init__()
        if low is not None:
            low = (low - mean) / sd
        if high is not None:
            high = (high - mean) / sd
        self.L, self.H, self.alpha = low, high, alpha
        
    def forward(self, t): 
        if self.L is not None:
            t[t < self.L] = self.alpha * (t[t < self.L] - self.L) + self.L
        if self.H is not None:
            t[t > self.H] = self.alpha * (t[t > self.H] - self.H) + self.H
        return t


class Holiday(nn.Module):
    def __init__(self, holiday, repeat_every=365, mean=0, scale=1):
        super().__init__()
        self.holiday = (holiday - mean) / scale
        self.repeat_every = repeat_every / scale
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


class LinearX(nn.Module):
    def __init__(self, dims):
        super().__init__()
        if dims > 0:
            self.linear = nn.Linear(dims, 1, bias=False)
        
    def forward(self, x):
        if x is not None:
            return self.linear(x)
        else:
            return 0
    

class DefaultModel(nn.Module):
    def __init__(self, moments, breakpoints=None, y_n=7, m_n=5, w_n=0, l=None, h=None):
        super().__init__()
        if 'x' in moments:
            dims = moments['x'][0].shape[1]
        else:
            dims = 0

        self.trend = Trend(breakpoints, moments)
        self.seasonal = Seasonal(y_n, m_n, w_n, scale=moments['t'][1])
        self.linear = LinearX(dims)
        self.squash = Squasher(l, h, *moments['y'])

    def forward(self, t, x=None):
        prediction = self.seasonal(t) + self.trend(t) + self.linear(x)
        prediction = self.squash(prediction)
        return prediction