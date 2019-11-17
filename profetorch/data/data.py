import torch
from torch.utils.data import Dataset, DataLoader
from fastai.data_block import DataBunch, DatasetType

__all__ = ['TimeSeries', 'create_db']

class TimeSeries(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x, self.y = x, y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        return torch.Tensor([self.x[i]]), torch.Tensor([self.y[i]])
    
def create_db(t, y, t_val=None, y_val=None, train_p=0.8, bs=96):
    if t_val is None:
        train_len = int(train_p*len(y))
        train_ds = TimeSeries(t[:train_len], y[:train_len])
        test_ds = TimeSeries(t[train_len:], y[train_len:])
    else:
        train_ds = TimeSeries(t, y)
        test_ds = TimeSeries(t_val, y_val)
    
    return DataBunch.create(train_ds, test_ds, bs=bs)