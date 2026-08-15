import torch

class CrossEntropyLoss:
    def forward(self, 
                logits:torch.Tensor,        # (N,C)
                targets:torch.Tensor        # (N,)
                ):
        self.cache = (logits, targets)
        logits_max = logits.max(dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
        self.probs = probs   

        N = logits.shape[0]
        loss = -torch.log(probs[range(N), targets] + 1e-8).mean()
        return loss

    def backward(self):
        logits, targets = self.cache
        N = logits.shape[0]
        
        grad = self.probs.clone()
        grad[range(N), targets] -= 1
        grad /= N
        return grad

class MSELoss:
    def forward(self, 
                pred:torch.Tensor, 
                target:torch.Tensor):
        self.cache = (pred, target)
        return ((pred - target) ** 2).mean()

    def backward(self):
        pred, target = self.cache
        return 2 * (pred - target) / pred.numel()   