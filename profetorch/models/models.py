import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai.basics import Learner

from .blocks import DefaultModel, DefaultQModel
from ..data.data import create_db, create_tensors, get_moments
from ..losses import q_loss

__all__ = ['Model']

class Model(nn.Module):
    def __init__(self, df, model=None, model_args=None, quantiles=[0.05, 0.5, 0.95], loss=None, bs=128, lr=0.3, wd=0):
        """
        parameters:
        - df: dataset used in training dataset.
        - model (optional): how to model time series. Default: DefaultModel.
        - loss (optional): loss function: Default l1 loss.
        - bs (optional): batchsize
        - wd (optional): weight decay
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
        self.wd = wd
        self.bs = bs
        
    def fit(self, df=None, epochs=20):
        # self.find_appropriate_lr(df)
        learner = self.create_learner(df)
        learner.fit(epochs, self.lr)

        self.model = learner.model
        
    def find_appropriate_lr(self, df, lr_diff:int = 15, loss_threshold:float = .05, adjust_value:float = 1, plot:bool = False) -> float:
        """
        Automatic LR finder. Credit https://forums.fast.ai/t/automated-learning-rate-suggester/44199
        """
        #Run the Learning Rate Finder
        model = self.create_learner(df)
        model.lr_find()
        
        #Get loss values and their corresponding gradients, and get lr values
        losses = np.array(model.recorder.losses)
        assert(lr_diff < len(losses))
        loss_grad = np.gradient(losses)
        lrs = model.recorder.lrs
        
        #Search for index in gradients where loss is lowest before the loss spike
        #Initialize right and left idx using the lr_diff as a spacing unit
        #Set the local min lr as -1 to signify if threshold is too low
        r_idx = -1
        l_idx = r_idx - lr_diff
        while (l_idx >= -len(losses)) and (abs(loss_grad[r_idx] - loss_grad[l_idx]) > loss_threshold):
            local_min_lr = lrs[l_idx]
            r_idx -= 1
            l_idx -= 1

        lr_to_use = local_min_lr * adjust_value
        self.lr = lr_to_use
        
        if plot:
            # plots the gradients of the losses in respect to the learning rate change
            plt.plot(loss_grad)
            plt.plot(len(losses)+l_idx, loss_grad[l_idx],markersize=10,marker='o',color='red')
            plt.ylabel("Loss")
            plt.xlabel("Index of LRs")
            plt.show()

            plt.plot(np.log10(lrs), losses)
            plt.ylabel("Loss")
            plt.xlabel("Log 10 Transform of Learning Rate")
            loss_coord = np.interp(np.log10(lr_to_use), np.log10(lrs), losses)
            plt.plot(np.log10(lr_to_use), loss_coord, markersize=10,marker='o',color='red')
            plt.show()
            
        # return lr_to_use
        
    def predict(self, df):
        x, _ = create_tensors(df.copy(), self.moments, predict=True)
        mean, sd = self.moments['y']
        y = sd * self.model(**x) + mean
        return y.detach().cpu().numpy()
    
    def forward(self, *args):
        return self.model(*args)
    
    def create_learner(self, df):
        db = create_db(df, bs=self.bs, moments=self.moments)
        learner = Learner(db, self.model, loss_func=self.loss, wd=self.wd)
        return learner
        