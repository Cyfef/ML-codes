import torch

class Linear:
    def __init__(self, 
                 in_dim:int, 
                 out_dim:int):
        # Kaiming init
        self.W = torch.randn(in_dim, out_dim) * (2.0 / in_dim)**0.5     # (in_dim,out_dim)
        self.b = torch.zeros(out_dim)          # (out_dim,)

        self.grad_W = torch.zeros_like(self.W)
        self.grad_b = torch.zeros_like(self.b)

        self.cache = None

    def forward(self, 
                x:torch.Tensor  # (*,in_dim)
                ):
        self.cache = x
        return x @ self.W + self.b  # (*,out_dim)

    def backward(self, 
                 dout:torch.Tensor  # (*, out_dim)
                 ):
        '''
        dout: upstream gradient
        '''

        x = self.cache

        dx = dout @ self.W.T

        self.grad_W.copy_(x.T @ dout)
        self.grad_b.copy_(dout.sum(axis=0))

        return dx   




    