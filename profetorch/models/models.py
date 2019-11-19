import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai.basics import Learner

from .blocks import Seasonal, Trend, DefaultModel
from ..data.data import create_db, convert_date

__all__ = ['Model']

class Model(nn.Module):
    def __init__(self, 
                 model=None, 
                 breakpoints=None, 
                 y_n=7, 
                 m_n=5, 
                 w_n=0, 
                 loss=F.l1_loss, 
                 bs=128, 
                 lr=0.5, 
                 wd=0):
        super().__init__()
        if model:
            self.model = model
        else:
            self.model = DefaultModel(breakpoints, y_n, m_n, w_n)
            
        self.loss = loss
        self.lr = lr
        self.wd = wd
        self.bs = bs
        
    def fit(self, df=None, t_train=None, y_train=None, t_val=None, y_val=None, epochs=20):
        db = create_db(df, t_train, y_train, t_val, y_val, bs=self.bs)
        learner = Learner(db, self.model, loss_func=self.loss)
        learner.fit(epochs, self.lr)
        
        if 'x' in db.train_ds.data: self.x = True
        self.model = learner.model
        
    def find_lr(self, df=None, t_train=None, y_train=None, t_val=None, y_val=None):
        # breakpoint()
        db = create_db(df, t_train, y_train, t_val, y_val, bs=self.bs)
        learner = Learner(db, self.model, loss_func=self.loss)
        learner.lr_find()
        learner.recorder.plot(skip_end=0)
        
    def predict(self, df):
        if df is not None:
            t = convert_date(df['date'].values)
            df.drop('date', axis=1, inplace=True)
            if df.shape[1] > 0:
                x = torch.Tensor(df.values)
                if len(x.shape) < 2:
                    x = x[:,None]
                return self.model(t, x)
            else:
                return self.model(t)
        
    def set_lr(self, lr): 
        self.lr = lr
    
    def forward(self, x):
        breakpoint()
        return self.model(x)
        