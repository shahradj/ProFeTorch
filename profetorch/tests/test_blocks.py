import torch
from ..models import *

blocks_ = [Trend, FourierModel, Seasonal, Squasher]

def test_hols():
    pass
    
def test_shape():
    x = torch.randn(10,1)
    for block in blocks_:
        m = block() # default parameters
        y = m(x)
        assert y.shape == (x.shape), f'output shape {y.shape} does not match input {x.shape}'
