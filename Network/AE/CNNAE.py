import torch
import torch.nn as nn
import torch.nn.functional as F

CNNAE_ENCODING_DIM = 64

class Encoder(nn.Module):
    def __init__(
            self, 
            encoding_dim:int
    ):
        '''
        CNN Encoder

        Args:
            encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        super().__init__()

        self.encoding_dim = encoding_dim

        # Conv blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 3 -> 32
        self.pool = nn.MaxPool2d(2, 2)                            # H, W -> H/2, W/2

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 -> 64
        # pool -> H/4, W/4

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 64 -> 128

        self.flatten_dim = 128 * 6 * 6 
        self.fc = nn.Linear(self.flatten_dim, self.encoding_dim)


    def forward(
            self, 
            x:torch.Tensor      #(Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
    ):
        '''
        The forward pass of the Encoder

        Args:
            x: input images        
        '''

        # Conv + ReLU + Pool
        x = F.relu(self.conv1(x))   # (B, 32, H, W)
        x = self.pool(x)            # (B, 32, H/2, W/2)

        x = F.relu(self.conv2(x))   # (B, 64, H/2, W/2)
        x = self.pool(x)            # (B, 64, H/4, W/4)

        x = F.relu(self.conv3(x))   # (B, 128, H/4, W/4)

        x = x.view(x.size(0), -1)   # Flatten
        v = self.fc(x)
        return v    #(Batch_size, encoding_dim)

class Decoder(nn.Module):
    def __init__(
            self, 
            encoding_dim:int
    ):
        '''
        CNN Encoder

        Args:
            encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        super().__init__()
        
        self.encoding_dim = encoding_dim

        self.flatten_dim = 128 * 6 * 6
        self.fc = nn.Linear(encoding_dim, self.flatten_dim)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # H/4 -> H/2
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)    # H/2 -> H

    def forward(
            self, 
            v:torch.Tensor       #(Batch_size, encoding_dim)
    ):
        '''
        The forward pass of the Decoder

        Args:
            v: latent vector
        '''
        
        x = self.fc(v)
        x = x.view(-1, 128, 6, 6)   # Reshape 

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x    #(Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)

class CNNAutoencoder(nn.Module):
    def __init__(
            self, 
            encoding_dim:int
    ):
        '''
        CNN Autoencoder

        Args:
            encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        super().__init__()
        self.encoder = Encoder(encoding_dim)
        self.decoder = Decoder(encoding_dim)

    def forward(
            self, 
            x
    ):
        '''
        The forward pass of the CNN Autoencoder
        '''
        
        v = self.encoder(x)
        x_recon = self.decoder(v)
        return x_recon
    
    @property
    def name(self):
        return "CNNAE"