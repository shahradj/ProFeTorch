import torch
from inspect import getmembers, isfunction
# import losses as loss
from ..losses import *

y1 = torch.arange(10, dtype=torch.float32)[:,None]
y2 = torch.arange(9, -1, -1, dtype=torch.float32)[:,None]

def test_zero_loss():
    loss_fns = [mse, mae, q_loss]
    for fn in loss_fns:
        assert fn(y1, y1) == 0, "Loss with self is not zero."

def test_one_loss():
    loss_fns = [mse, mae]
    for fn in loss_fns:
        assert fn(y1, y1+1) == 1, f"Mean loss of {fn.__name__} should be 1."