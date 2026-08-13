class CrossEntropyLoss:
    def forward(self, logits, targets):
        # logits: (N, C), targets: (N,) 类别索引
        self.cache = (logits, targets)
        # 数值稳定性：减去最大值
        logits_max = logits.max(dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
        self.probs = probs   # 保存，便于梯度计算
        N = logits.shape[0]
        loss = -torch.log(probs[range(N), targets] + 1e-8).mean()
        return loss

    def backward(self):
        # 返回损失对 logits 的梯度
        logits, targets = self.cache
        N = logits.shape[0]
        grad = self.probs.clone()
        grad[range(N), targets] -= 1
        grad /= N
        return grad