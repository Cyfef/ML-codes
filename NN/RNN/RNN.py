import torch
from wandb_utils import *

class AffineLayer:
    """全连接层"""
    def __init__(self, 
                 in_dim:int, 
                 out_dim:int, 
                 dtype=torch.float32):
        self.W = torch.randn(in_dim, out_dim, dtype=dtype) / (in_dim ** 0.5)    # (in_dim, out_dim)
        self.b = torch.zeros(out_dim, dtype=dtype)      # (out_dim,)

        self.cache = None

    def forward(self, 
                x:torch.Tensor  # (...,in_dim)
                ):

        self.cache = x  # save shape
        in_dim = self.W.shape[0]

        out = x.reshape(-1, in_dim) @ self.W + self.b
        
        out_shape = x.shape[:-1] + (self.W.shape[1],)
        return out.reshape(out_shape)


    def backward(self, 
                 dout:torch.Tensor
                 ):

        x = self.cache
        in_dim, out_dim = self.W.shape
        
        x_flat = x.reshape(-1, in_dim)
        dout_flat = dout.reshape(-1, out_dim)
        
        dx_flat = dout_flat @ self.W.T
        dW = x_flat.T @ dout_flat
        db = dout_flat.sum(dim=0)
        
        dx = dx_flat.reshape(x.shape)
        
        return dx, dW, db


class WordEmbeddingLayer:
    def __init__(self, 
                 V:int, 
                 wordvec_dim:int, 
                 dtype=torch.float32):
        # Embedding matrix
        self.W = torch.randn(V, wordvec_dim, dtype=dtype) / 100     # (V,wordvec_dim)
        self.cache = None

    def forward(self, 
                x:torch.Tensor      # (N,T)
                ):
        '''x: word index int'''
        self.cache = x
        return self.W[x]   # (N, T, wordvec_dim)

    def backward(self, 
                 dout:torch.Tensor  # (N, T, wordvec_dim)
                 ):
        """
        dout: (N, T, wordvec_dim)
        返回: dW (V, wordvec_dim)
        """
        x = self.cache
        dW = torch.zeros_like(self.W)

        N, T = x.shape

        x_flat = x.reshape(N * T)          # (N*T,)
        dout_flat = dout.reshape(N * T, -1) # (N*T, wordvec_dim)
    
        dW.index_add_(0, x_flat, dout_flat)
        return dW   # (V, wordvec_dim)


class VanillaRNNCell:
    def __init__(self, 
                 x_dim:int, 
                 h_dim:int, 
                 dtype=torch.float32):
        self.Wx = torch.randn(x_dim, h_dim, dtype=dtype) / (x_dim ** 0.5)   # (x_dim,h_dim)
        self.Wh = torch.randn(h_dim, h_dim, dtype=dtype) / (h_dim ** 0.5)   # (h_dim,h_dim)
        self.b = torch.zeros(h_dim, dtype=dtype)                            # (h_dim,)
        self.cache = None

    def step_forward(self, 
                     x:torch.Tensor,    # (N,x_dim) 
                     prev_h:torch.Tensor    # (N,h_dim)
                     ):

        next_h = torch.tanh(prev_h @ self.Wh + x @ self.Wx + self.b)    # (N,h_dim)
        self.cache = (next_h, prev_h, x)
        return next_h

    def step_backward(self, 
                      dnext_h:torch.Tensor  # (N,h_dim)
                      ):

        next_h, prev_h, x = self.cache

        dh = dnext_h * (1 - next_h ** 2)   # (N, h_dim)
        db = dh.sum(dim=0)                 # (h_dim,)
        dprev_h = dh @ self.Wh.T           # (N, h_dim)
        dWh = prev_h.T @ dh                # (h_dim, h_dim)
        dx = dh @ self.Wx.T                # (N, x_dim)
        dWx = x.T @ dh                     # (x_dim, h_dim)

        return dx, dprev_h, dWx, dWh, db


class RNNLoop:
    def __init__(self, 
                 cell:VanillaRNNCell):
        self.cell = cell
        self.caches = []   

    def forward(self, 
                x:torch.Tensor,     # (N, T, x_dim)
                h0:torch.Tensor     # (N, h_dim)
                ):

        N, T, _ = x.shape
        h = h0

        outputs = []    # (h1,h2,...,hT)
        self.caches = []

        for t in range(T):
            h = self.cell.step_forward(x[:, t, :], h)   # (N,h_dim)
            outputs.append(h)
            self.caches.append(self.cell.cache)   
        return torch.stack(outputs, dim=1)   # (N, T, h_dim)

    def backward(self, 
                 dh:torch.Tensor    # (N, T, h_dim)
                 ):
 
        N, T, h_dim = dh.shape

        first_cache = self.caches[0]
        _, prev_h0, x0 = first_cache
        x_dim = x0.shape[1]   

        dx = torch.zeros(N, T, x_dim, dtype=dh.dtype, device=dh.device)     # (N, T, x_dim)
        dWx = torch.zeros_like(self.cell.Wx)                                # (x_dim, h_dim)
        dWh = torch.zeros_like(self.cell.Wh)                                # (h_dim, h_dim)
        db = torch.zeros(h_dim, dtype=dh.dtype, device=dh.device)           # (h_dim,)
        dprev_h = torch.zeros(N, h_dim, dtype=dh.dtype, device=dh.device)   # (N, h_dim)

        for t in reversed(range(T)):
            dnext_h = dh[:, t, :] + dprev_h   

            self.cell.cache = self.caches[t]   
            dxt, dprev_h, dWxt, dWht, dbt = self.cell.step_backward(dnext_h)

            dx[:, t, :] = dxt
            dWx += dWxt
            dWh += dWht
            db += dbt

        dh0 = dprev_h   
        return dx, dh0, dWx, dWh, db


def temporal_softmax_loss(logits:torch.Tensor,        # (N,T,V)
                          labels:torch.Tensor,        # (N,T)
                          mask:torch.Tensor      # (N,T)
                          ):
    """
    logits: (N, T, V) logits 
    labels: (N, T) 真实标签（整数）
    mask: (N, T) 布尔有效标记padding
    """
    N, T, V = logits.shape

    logits_flat = logits.reshape(N * T, V)
    labels_flat = labels.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    logits_max = logits_flat.max(dim=1, keepdim=True)[0]
    exp_logits = torch.exp(logits_flat - logits_max)
    probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)

    loss = -torch.sum(mask_flat * torch.log(probs[torch.arange(N * T), labels_flat] + 1e-8)) / N

    dlogits_flat = probs.clone()
    dlogits_flat[torch.arange(N * T), labels_flat] -= 1
    dlogits_flat /= N
    dlogits_flat *= mask_flat.unsqueeze(1)  

    dlogits = dlogits_flat.reshape(N, T, V)
    return loss, dlogits


class CaptioningRNN:
    def __init__(
        self,
        word_to_idx:dict,
        img_feat_dim:int=512,
        wordvec_dim:int=128,
        h_dim:int=128,
        dtype=torch.float32,
    ):
        
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}   # reverse dict

        self.dtype = dtype

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        V = len(word_to_idx)

        self.embedding = WordEmbeddingLayer(V, wordvec_dim, dtype)
        self.projection = AffineLayer(img_feat_dim, h_dim, dtype)

        self.rnn_cell = VanillaRNNCell(wordvec_dim, h_dim, dtype)
        self.rnn = RNNLoop(self.rnn_cell)

        self.vocabulary = AffineLayer(h_dim, V, dtype)

        self.params = {
            'W_embed': self.embedding.W,
            'W_proj': self.projection.W,
            'b_proj': self.projection.b,
            'Wx': self.rnn_cell.Wx,
            'Wh': self.rnn_cell.Wh,
            'b': self.rnn_cell.b,
            'W_vocab': self.vocabulary.W,
            'b_vocab': self.vocabulary.b,
        }

    def loss(self, 
             features:torch.Tensor,     # (N,img_feat_dim) 
             captions:torch.Tensor      # (N,T+1) 
             ):
        """
        features: (N, D)
        captions: (N, T+1) 包括 <START>
        返回: (loss 标量, grads 字典)
        """
        captions_in = captions[:, :-1]          # (N, T)
        captions_out = captions[:, 1:]          # (N, T)

        mask = (captions_out != self._null)     # (N, T)

        # forward
        h0 = self.projection.forward(features)          # (N, h_dim)
        x = self.embedding.forward(captions_in)         # (N, T, x_dim)
        h = self.rnn.forward(x, h0)                     # (N, T, h_dim)
        scores = self.vocabulary.forward(h)             # (N, T, V)

        # loss
        loss, dscores = temporal_softmax_loss(scores, captions_out, mask)

        # backward
        # 1) vocabulary
        dh, dW_vocab, db_vocab = self.vocabulary.backward(dscores)

        # 2) RNN
        dx, dh0, dWx, dWh, db = self.rnn.backward(dh)

        # 3) embedding
        dW_embed = self.embedding.backward(dx)

        # 4) projection
        _, dW_proj, db_proj = self.projection.backward(dh0)

        # grad dict
        grads = {
            'W_embed': dW_embed,
            'W_proj': dW_proj,
            'b_proj': db_proj,
            'Wx': dWx,
            'Wh': dWh,
            'b': db,
            'W_vocab': dW_vocab,
            'b_vocab': db_vocab,
        }
        return loss.item(), grads   

    def sample(self, 
               features:torch.Tensor,   # (N,img_feat_dim)
               max_length=30)->torch.Tensor:
        """
        test，返回 (N, max_length) 整数索引（不含 <START>）
        """
        N = features.shape[0]
        captions = torch.full((N, max_length), self._null, dtype=torch.long)

        h = self.projection.forward(features)   # (N, h_dim)
        word = torch.full((N,), self._start, dtype=torch.long)

        with torch.no_grad():   
            for t in range(max_length):
                x = self.embedding.forward(word.unsqueeze(1))[:, 0, :]   # (N, wordvec_dim)
                h = self.rnn_cell.step_forward(x, h)                     # (N, h_dim)

                scores = self.vocabulary.forward(h)                      # (N, V)
                word = scores.argmax(dim=1)                              # (N,)
                captions[:, t] = word
        return captions

    def to(self, device):

        for name, param in self.params.items():
            self.params[name] = param.to(device)

        self.embedding.W = self.params['W_embed']
        self.projection.W = self.params['W_proj']
        self.projection.b = self.params['b_proj']
        self.rnn_cell.Wx = self.params['Wx']
        self.rnn_cell.Wh = self.params['Wh']
        self.rnn_cell.b = self.params['b']
        self.vocabulary.W = self.params['W_vocab']
        self.vocabulary.b = self.params['b_vocab']
        return self


class CaptioningRNNTrainer():
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 device):
        
        self.model=model
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.device=device

    def train(self,
              train_dataloader,
              save_path:str,
              num_epochs: int = 10,
              log_interval: int = 50,
              ):

        wandb_init()
        iter_count=0

        for epoch in range(1,num_epochs+1):
            for (features, captions) in train_dataloader:
                
                features = features.to(self.device, dtype=torch.float32)
                captions = captions.to(self.device, dtype=torch.long)

                loss, grads = self.model.loss(features, captions)

                self.optimizer.zero_grad()

                for name, param in self.model.params.items():
                    if name in grads:
                        param.grad = grads[name]

                self.optimizer.step()

                wandb_log({
                            "train/loss": loss,
                            "train/iteration": iter_count,
                            })

                if iter_count % log_interval == 0:
                    print(f'Iter: {iter_count}, Loss: {loss:.4}')

                iter_count += 1

            self.scheduler.step()

        torch.save(self.model.params, save_path)
        wandb_finish()
