import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai.basics import Learner, LearnerCallback

from .blocks import DefaultModel, DefaultQModel
from ..data.data import create_db, create_tensors, get_moments
from ..losses import q_loss

__all__ = ['Model']
    
class L1Loss(LearnerCallback):
    def __init__(self, learn, beta=1e-2):
        super().__init__(learn)
        self.beta = beta
    
    def on_backward_end(self, **kwargss):
        weights = {k:v for k,v in self.learn.model.named_parameters() 
                   if not 'bias' in k}
        
        for k, v in weights.items():
            sign = torch.ones_like(v)
            sign[v<0] = -1
            v = v - self.beta * sign
            self.learn.model.state_dict()[k].copy_(v)


class Model(nn.Module):
    def __init__(self, df, model=None, model_args=None, quantiles=[0.05, 0.5, 0.95], loss=None, bs=128, lr=0.3, alpha=0, beta=0.1):
        """
        parameters:
        - df: dataset used in training dataset.
        - model (optional): how to model time series. Default: DefaultModel.
        - loss (optional): loss function: Default l1 loss.
        - bs (optional): batchsize
        - alpha (optional): weight decay
        """
        super().__init__()
        self.moments = get_moments(df.copy())

        if loss is None:
            self.model = DefaultQModel(self.moments, **model_args, quantiles=quantiles)
            self.loss = partial(q_loss, quantiles=quantiles)
        else:
            self.model = DefaultModel(self.moments, **model_args)
            self.loss = loss
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.bs = bs
        
    def fit(self, df=None, epochs=20):
        # self.find_appropriate_lr(df)
        learner = self.create_learner(df)
        cb = L1Loss(learner, self.beta)
        # breakpoint()
        
        learner.fit_one_cycle(epochs, self.lr, callbacks=cb)

        self.model = learner.model
        
    def predict(self, df):
        x, _ = create_tensors(df.copy(), self.moments, predict=True)
        mean, sd = self.moments['y']
        y = sd * self.model(**x) + mean
        return y.detach().cpu().numpy()
    
    def forward(self, *args):
        return self.model(*args)
    
    def create_learner(self, df):
        db = create_db(df, bs=self.bs, moments=self.moments)
        learner = Learner(db, self.model, loss_func=self.loss, wd=self.alpha)
        return learner
        