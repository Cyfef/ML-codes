import torch

class ReLU:
    def forward(self, 
                x:torch.Tensor):
        self.cache = x
        return torch.maximum(x, torch.tensor(0.0))

    def backward(self, 
                 dout:torch.Tensor):
        x = self.cache
        dx = dout.clone()
        dx[x <= 0] = 0
        return dx


class Softmax:
    def __init__(self, 
                 dim:int=1):
        self.dim = dim          

    def forward(self, 
                x:torch.Tensor):
        self.cache = x
   
        x_shifted = x - x.max(dim=self.dim, keepdim=True)[0]
        exp_x = torch.exp(x_shifted)
        self.probs = exp_x / exp_x.sum(dim=self.dim, keepdim=True)
        return self.probs

    def backward(self, 
                 dout:torch.Tensor):
        y = self.probs
        sum_y_dout = (y * dout).sum(dim=self.dim, keepdim=True)
        dx = y * (dout - sum_y_dout)
        return dx