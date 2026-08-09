import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, 
                 noise_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(noise_dim, 1024),     # noise_dim -> 1024
            nn.ReLU(),
            nn.Linear(1024, 1024),          # 1024 -> 1024
            nn.ReLU(),
            nn.Linear(1024, 784),           # 1024 -> 784
            nn.Tanh()                       # [-1, 1]
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(784, 256),          # 784 -> 256
            nn.LeakyReLU(0.01),           # alpha=0.01
            nn.Linear(256, 256),          # 256 -> 256
            nn.LeakyReLU(0.01),           # alpha=0.01
            nn.Linear(256, 1)             # 256 -> 1  logits
        )

    def forward(self, x):
        return self.net(x)



class GANTrainer:
    def __init__(self, 
                 generator, 
                 discriminator, 
                 noise_dim, 
                 lr=1e-3, 
                 betas=(0.5, 0.999), 
                 dtype=float,
                 device='cpu'):
        
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)

        self.noise_dim = noise_dim
        self.dtype=dtype
        self.device = device

        self.D_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.G_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)

    def train(self, 
              train_dataloader, 
              num_epochs, 
              log_interval=250):
        
        iter_count=0
        for epoch in range(1, num_epochs + 1):
            for x, _ in train_dataloader:
                batch_size = x.shape[0]
            
                # discriminator optim
                self.D_optimizer.zero_grad()
            
                # real
                real_data=x.view(-1,784).to(self.device)
                logits_real=self.discriminator(2* (real_data - 0.5))
            
                # fake
                noise_z = 2 * torch.rand(batch_size, self.noise_dim, dtype=real_data.dtype, device=real_data.device) - 1
                fake_imgs=self.generator(noise_z).detach()
                logits_fake=self.discriminator(fake_imgs)
            
                D_loss=self.discriminator_loss(logits_real,logits_fake)
                D_loss.backward()
                self.D_optimizer.step()
            
                # generator optim
                self.G_optimizer.zero_grad()
            
                # fake
                noise_z = 2 * torch.rand(batch_size, self.noise_dim, dtype=real_data.dtype, device=real_data.device) - 1
                fake_imgs=self.generator(noise_z)
                logits_fake=self.discriminator(fake_imgs)
            
                G_loss=self.generator_loss(logits_fake)
                G_loss.backward()
                self.G_optimizer.step()
            
                if iter_count % log_interval == 0:
                    print(f'Iter: {iter_count}, D: {D_loss.item():.4}, G:{G_loss.item():.4}')
            
                iter_count += 1

    @staticmethod
    def discriminator_loss(logits_real, logits_fake):
        """
        logits_real: (N,) 真实样本得分
        logits_fake: (N,) 生成样本得分
        """
        real_loss = F.binary_cross_entropy_with_logits(
            logits_real, torch.ones_like(logits_real), reduction='mean'
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            logits_fake, torch.zeros_like(logits_fake), reduction='mean'
        )
        return real_loss + fake_loss

    @staticmethod
    def generator_loss(logits_fake):
        return F.binary_cross_entropy_with_logits(
            logits_fake, torch.ones_like(logits_fake), reduction='mean'
        )
