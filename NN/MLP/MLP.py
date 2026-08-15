import torch
from func import ReLU
from nn import Linear


class MLP:
    def __init__(self):
        self.fc1 = Linear(1, 64)
        self.relu1 = ReLU()
        self.fc2 = Linear(64, 64)
        self.relu2 = ReLU()
        self.fc3 = Linear(64, 1)

        self.params = [
            {'param': self.fc1.W, 'grad': self.fc1.grad_W},
            {'param': self.fc1.b, 'grad': self.fc1.grad_b},
            {'param': self.fc2.W, 'grad': self.fc2.grad_W},
            {'param': self.fc2.b, 'grad': self.fc2.grad_b},
            {'param': self.fc3.W, 'grad': self.fc3.grad_W},
            {'param': self.fc3.b, 'grad': self.fc3.grad_b},
        ]

    def forward(self, 
                x:torch.Tensor):
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        x = self.fc3.forward(x)
        return x

    def backward(self, 
                 dout:torch.Tensor):
        dout = self.fc3.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.fc2.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.fc1.backward(dout)
        return dout