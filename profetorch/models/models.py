import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai.basics import Learner

from .blocks import Seasonal, Trend, DefaultModel
from ..data.data import create_db

__all__ = ['Model']

class Model(nn.Module):
    def __init__(self, model=None, breakpoints=None, y_n=7, m_n=5, w_n=0, loss=F.l1_loss, lr=0.5, wd=0):
        super().__init__()
        if model:
            self.model = model
        else:
            self.model = DefaultModel(breakpoints, y_n, m_n, w_n)
            
        self.loss = loss
        self.lr = lr
        self.wd = wd
        
    def fit(self, t_train, y_train, t_val=None, y_val=None, epochs=20):
        db = create_db(t_train, y_train, t_val, y_val)
        learner = Learner(db, self.model, loss_func=self.loss)
        learner.fit(epochs, self.lr)
        self.model = learner.model
        
    def find_lr(self, t_train, y_train, t_val=None, y_val=None):
        db = create_db(t_train, y_train, t_val, y_val)
        learner = Learner(db, self.model, loss_func=self.loss)
        learner.lr_find()
        learner.recorder.plot(skip_end=0)
        
    def set_lr(self, lr): 
        self.lr = lr
    
    def forward(self, x):
        return self.model(x)
        