import torch
from fastai.basics import LearnerCallback

__all__ = ['L1Loss', 'PrintLoss']

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
           
def get_loss(model, dl, loss_func):
    "Calculate `loss_func` of `model` on `dl` in evaluation mode."
    model.eval()
    with torch.no_grad():
        loss = 0
        N = 0
        for xb,yb in dl:
            # breakpoint()
            if isinstance(xb, list):
                loss += loss_func(model(*xb), yb) * len(yb)
            else:
                loss += loss_func(model(xb), yb) * len(yb)
            N += len(yb)
        return loss / N
                    
class PrintLoss(LearnerCallback):
    def __init__(self, learn):
        super().__init__(learn)
        
    def on_epoch_end(self, **kwargs):
        train_loss = get_loss(self.learn.model, self.learn.data.train_dl, loss_func=self.learn.loss_func)
        val_loss = get_loss(self.learn.model, self.learn.data.valid_dl, loss_func=self.learn.loss_func)
        
        epoch = kwargs['epoch']
        n_epochs = kwargs['n_epochs']
        
        print(f'Epoch {epoch+1}/{n_epochs} Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}', end='\r')