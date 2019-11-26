import torch
import torch.nn as nn
import torch.nn.functional as F

def mse(y_pred, y, weights=None):
    """
    Mean Squared Error
    """
    return weighted_loss((y_pred - y)**2, weights)

def mae(y_pred, y, weights=None):
    """
    Mean Absolute Error
    """
    return weighted_loss(torch.abs(y_pred - y), weights)
    
def q_loss(y_pred, y, quantiles = [0.05, 0.5, 0.95], weights=None):
    """
    y_pred: Predicted Value
    y: Target
    quantiles: Quantile
    weights(optional): Weighting of prediction-target pair.
    """
    if isinstance(quantiles, list):
        e = 0
        for q, pred in zip(quantiles, y_pred.split(1, dim=-1)):
            e += weighted_loss(tilted_loss(pred, y, q), weights)   
        return e.mean()
    else:
        return weighted_loss(tilted_loss(y_pred, y, quantiles), weights)

def tilted_loss(y_pred, y, q=0.5):
    e = (y - y_pred)
    return q * torch.clamp_min(e, 0) + (1-q) * torch.clamp_min(-e, 0)
        
def weighted_loss(loss, weights):
    """
    Weighted loss
    """
    if weights is not None:
        return torch.mean(weights * loss)
    else:
        return torch.mean(loss)
