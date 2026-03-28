import torch
from torch import nn
from torch.nn import functional as F

VAE_ENCODING_DIM = 64

# Define the Variational Encoder
class VarEncoder(nn.Module):
    def __init__(self, encoding_dim):
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        super(VarEncoder, self).__init__()
        # TODO: implement the encoder
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # 128 * 6 * 6 = 4608
        self.fc_mu = nn.Linear(4608, encoding_dim)
        self.fc_logvar = nn.Linear(4608, encoding_dim)


    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        return log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        '''
        
        # TODO: implement the forward 
        x = F.relu(self.conv1(x))   # 32x24x24
        x = self.pool(x)            # 32x12x12

        x = F.relu(self.conv2(x))   # 64x12x12
        x = self.pool(x)            # 64x6x6

        x = F.relu(self.conv3(x))   # 128x6x6

        x = x.view(x.size(0), -1)   # 4608

        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        
        return mu, log_var

# Define the Decoder
class VarDecoder(nn.Module):
    def __init__(self, encoding_dim):
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        super(VarDecoder, self).__init__()
        # TODO: implement the decoder
        self.fc = nn.Linear(encoding_dim, 4608)

        self.deconv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.deconv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.deconv3 = nn.Conv2d(32, 3, 3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, v):
        '''
        v: latent vector, dim: (Batch_size, encoding_dim)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        '''
        
        # TODO: implement the forward pass
        x = self.fc(v)

        x = x.view(v.size(0), 128, 6, 6)

        x = F.relu(self.deconv1(x))   # 64x6x6

        x = self.upsample(x)          # 64x12x12
        x = F.relu(self.deconv2(x))   # 32x12x12

        x = self.upsample(x)          # 32x24x24
        x = torch.sigmoid(self.deconv3(x))  # 3x24x24

        return x

# Define the Variational Autoencoder
class VarAutoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(VarAutoencoder, self).__init__()
        self.encoder = VarEncoder(encoding_dim)
        self.decoder = VarDecoder(encoding_dim)

    @property
    def name(self):
        return "VAE"

    def reparameterize(self, mu, log_var):
        '''
        mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        return v: sampled latent vector, dim: (Batch_size, encoding_dim)
        '''
        
        
        # TODO: implement the reparameterization trick to sample v
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
        
    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        return log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        '''
        # TODO: implement the forward pass
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var
        return x, mu, log_var

# Loss Function
def VAE_loss_function(outputs, images):
    '''
    outputs: (x, mu, log_var)
    images: input/original images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
    return loss: the loss value, dim: (1)
    '''
    # TODO: implement the loss function for VAE
    x_recon, mu, log_var = outputs

    # Reconstruction Loss
    recon_loss = F.mse_loss(x_recon, images, reduction='sum')

    # KL Divergence
    kl_loss = -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp()
    )

    return recon_loss + kl_loss

