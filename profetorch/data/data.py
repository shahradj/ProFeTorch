import torch
from torch.utils.data import Dataset, DataLoader
from fastai.data_block import DataBunch, DatasetType

__all__ = ['create_db']

class TimeSeries(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x, self.y = x, y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        return torch.Tensor([self.x[i]]), torch.Tensor([self.y[i]])
    
def convert_date(dates):
    MILISECONDS_IN_DAY = 1e9*3600*24
    dates = dates.astype(int) / MILISECONDS_IN_DAY
    return torch.Tensor(dates).squeeze()[:,None]

class DataFrame(Dataset):
    def __init__(self, df):
        super().__init__()
        t = convert_date(df['date'].values)
        y = torch.Tensor(df['y'].values).squeeze()[:,None]
        df.drop(['date', 'y'], axis=1, inplace=True)
        
        vars = {'t': t, 'y': y}
        if df.shape[1] > 0:
            x = torch.Tensor(df.values).float()
            vars['x'] = x
          
        self.data = vars
        # if moments is None:
        #     moments = {k: [v.mean(), v.std()] for k,v in vars}
        # self.moments = moments
        
        # self.data = {k: (vars[k] - mean) / std for k, (mean, std) in moments}
    
    def __len__(self):
        return len(self.data['t'])
    
    def __getitem__(self, i):
        if 'x' in self.data:
            return (self.data['t'][i], self.data['x'][i]), self.data['y'][i]
        else:
            return self.data['t'][i], self.data['y'][i]

    
def create_db(df=None, t=None, y=None, t_val=None, y_val=None, train_p=0.8, bs=96):
    if df is None:
        if t_val is None:
            train_len = int(train_p*len(y))
            train_ds = TimeSeries(t[:train_len], y[:train_len])
            val_ds = TimeSeries(t[train_len:], y[train_len:])
        else:
            train_ds = TimeSeries(t, y)
            val_ds = TimeSeries(t_val, y_val)
    else:
        train_len = int(train_p*len(df))
        df.reset_index(drop=True, inplace=True)
        train_ds = DataFrame(df.iloc[:train_len])
        val_ds = DataFrame(df.iloc[train_len:])
    
    bs = min(bs, len(train_ds))
    val_bs = min(bs, len(val_ds))
    # breakpoint()
    return DataBunch.create(train_ds, val_ds, bs=bs, val_bs=val_bs)