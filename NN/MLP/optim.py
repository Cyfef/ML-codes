import torch
from typing import List,Tuple

class SGD:
    def __init__(self, 
                 params:List[dict], 
                 lr:float=0.01, 
                 momentum:float=0.0, 
                 weight_decay:float=0.0):
        
        self.params = params  # [{'param': W, 'grad': grad_W}] 
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [torch.zeros_like(p['param']) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            grad = p['grad'] + self.weight_decay * p['param']   # L2 
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
            p['param'] += self.velocities[i]

    def zero_grad(self):
        for p in self.params:
            p['grad'].zero_()

class RMSprop:
    def __init__(self, 
                 params:List[dict], 
                 lr:float=0.001, 
                 beta:float=0.9, 
                 eps:float=1e-8, 
                 weight_decay:float=0.0):
        self.params = params        
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay
        self.v = [torch.zeros_like(p['param']) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            grad = p['grad']
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p['param']

            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * (grad ** 2)
            p['param'] -= self.lr * grad / (torch.sqrt(self.v[i]) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p['grad'].zero_()


class Adam:
    def __init__(self, 
                 params:List[dict], 
                 lr:float=0.001, 
                 betas:Tuple[float]=(0.9, 0.999), 
                 eps:float=1e-8, 
                 weight_decay:float=0.0):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [torch.zeros_like(p['param']) for p in params]
        self.v = [torch.zeros_like(p['param']) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            grad = p['grad'] + self.weight_decay * p['param']

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            
            p['param'] -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p['grad'].zero_()