import torch
import unittest
from ..models import *

blocks_ = [Trend, FourierModel, Seasonal, Squasher]

class TestShape(unittest.TestCase):
    def test_shape(self):
        x = torch.randn(10,1)
        for block in blocks_:
            m = block() # default parameters
            y = m(x)
            self.assertEqual(y.shape, x.shape, f'output shape {y.shape} does not match input {x.shape}')
            
class TestHoliday(unittest.TestCase):
    def test_hols(self):
        pass
    