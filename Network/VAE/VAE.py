import torch
from torch import nn
from torch.nn import functional as F

VAE_ENCODING_DIM = 64

class VarEncoder(nn.Module):
    def __init__(
            self, 
            encoding_dim:int
        ):
        '''
        Args:
            encoding_dim: Dimension of latent space (z)
        '''
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)     # (B,3,24,24) → (B,32,24,24)
        self.pool = nn.MaxPool2d(2, 2)                  # downsample by 2

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    # (B,32,12,12) → (B,64,12,12)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)   # (B,64,6,6) → (B,128,6,6)

        # 128 * 6 * 6 = 4608
        self.fc_mu = nn.Linear(4608, encoding_dim)      # mean
        self.fc_logvar = nn.Linear(4608, encoding_dim)  # log variance

    def forward(
            self, 
            x:torch.Tensor      #(Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        ):
        '''
        Args:
            x: input images              
        '''

        x = F.relu(self.conv1(x))   # 32x24x24
        x = self.pool(x)            # 32x12x12

        x = F.relu(self.conv2(x))   # 64x12x12
        x = self.pool(x)            # 64x6x6

        x = F.relu(self.conv3(x))   # 128x6x6

        x = x.view(x.size(0), -1)   # 4608

        mu = self.fc_mu(x)          #(Batch_size, encoding_dim)
        log_var = self.fc_logvar(x) #(Batch_size, encoding_dim)
        
        return mu, log_var

class VarDecoder(nn.Module):
    def __init__(
            self, 
            encoding_dim:int
        ):
        '''
        Args:
            encoding_dim: Dimension of latent space (z)
        '''
        super().__init__()
      
        self.fc = nn.Linear(encoding_dim, 4608)

        self.deconv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.deconv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.deconv3 = nn.Conv2d(32, 3, 3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(
            self, 
            z:torch.Tensor      #(Batch_size, encoding_dim)
        ):
        '''
        Args:
            z: latent vector   
        '''

        x = self.fc(z)

        x = x.view(z.size(0), 128, 6, 6)

        x = F.relu(self.deconv1(x))   # 64x6x6

        x = self.upsample(x)          # 64x12x12
        x = F.relu(self.deconv2(x))   # 32x12x12

        x = self.upsample(x)          # 32x24x24
        x = torch.sigmoid(self.deconv3(x))  # 3x24x24

        return x    #(Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)

class VAE(nn.Module):
    def __init__(
            self, 
            encoding_dim:int
        ):
        '''
        Variational Autoencoder (VAE)

        Args:
            encoding_dim: Dimension of latent space (z)
        '''
        super().__init__()
        self.encoder = VarEncoder(encoding_dim)
        self.decoder = VarDecoder(encoding_dim)

    def reparameterize(
            self, 
            mu:torch.Tensor,        #(Batch_size, encoding_dim)
            log_var:torch.Tensor    #(Batch_size, encoding_dim)
        ):
        '''
        Reparameterization Trick

        Args:
            mu: mean of the distribution
            log_var: log of the variance of the distribution 
        '''

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z    #(Batch_size, encoding_dim)
        
    def forward(
            self, 
            x:torch.Tensor       #(Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        ):
        '''
        Args:
            x: input images
        '''

        mu, log_var = self.encoder(x)           #(Batch_size, encoding_dim);(Batch_size, encoding_dim)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)               #(Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return x_recon, mu, log_var
    
    @property
    def name(self):
        return "VAE"

def VAE_loss_function(
        outputs, 
        images:torch.Tensor     #(Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
    ):
    '''
    VAE Loss Function

    Args:
        outputs: tuple (x_recon, mu, log_var)
        images: original images 
    '''

    x_recon, mu, log_var = outputs

    # Reconstruction Loss
    recon_loss = F.mse_loss(x_recon, images, reduction='sum')

    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kl_loss