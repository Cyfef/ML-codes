import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()

        self.block=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3),  # (B,32,30,30)
            nn.ReLU(),
            nn.Conv2d(32,32,kernel_size=3),  # (B,32,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),   # (B,32,14,14)

            nn.Conv2d(32,64,kernel_size=3),  # (B,64,12,12)
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3),  # (B,64,10,10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),   # (B,64,5,5)

            nn.Flatten(),   #(B,1600)

            nn.Linear(1600,512), # (B,512)
            nn.ReLU(),
            nn.Linear(512,num_class),  # (B,num_class)
        )

    def forward(self, 
                x:torch.Tensor
        )->torch.Tensor:
        return self.block(x)


class ConvNetTrainer():
    def __init__(self,
                 model,
                 optimizer,
                 dtype,
                 device):
        
        self.model=model.to(device)
        self.optimizer=optimizer

        self.dtype=dtype
        self.device = device
        

    def train(self,
              num_epochs:int,
              train_dataloader,
              log_interval:int=50
              ):
        self.model.train()
        iter_count=0

        for epoch in range(1,num_epochs+1):
            for imgs,labels in train_dataloader:
                batch_size=labels.shape[0]

                imgs=imgs.to(self.device)
                labels=labels.to(self.device)

                self.optimizer.zero_grad()

                logits=self.model(imgs)
                loss=self.CELoss(logits,labels)

                loss.backward()
                self.optimizer.step()

                if iter_count % log_interval == 0:
                    print(f'Iter: {iter_count}, Loss: {loss.item():.4}')

                iter_count += 1


    @staticmethod
    def CELoss(pred:torch.Tensor,   # (B,num_class)
               gt:torch.Tensor      # (B,)
        )->torch.Tensor:
        '''
        Cross-Entropy loss
        '''
        batch_size = pred.size(0)

        max_vals = torch.max(pred, dim=1, keepdim=True)[0]  # (B,1)
        log_sum_exp = torch.log(torch.sum(torch.exp(pred - max_vals), dim=1, keepdim=True)) + max_vals
        log_probs = pred - log_sum_exp  # (B,num_class)
        
        loss = -log_probs[torch.arange(batch_size), gt]  # (B,)
        return loss.mean()  # (1,)
    
