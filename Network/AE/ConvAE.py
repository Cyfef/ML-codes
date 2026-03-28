import torch
import torch.nn as nn
import torch.nn.functional as F

AE_ENCODING_DIM = 64

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Encoder, self).__init__()
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        
        '''
        TODO: Implement the Encoder.

        Requirements:
        1. Use convolutional layers to extract features from the input images.
        2. Apply max pooling to downsample the spatial dimensions.
        3. Use a linear layer to map the feature maps to the latent vector.
        '''
        # Conv blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 3 -> 32
        self.pool = nn.MaxPool2d(2, 2)                            # H, W -> H/2, W/2

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 -> 64
        # pool -> H/4, W/4

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 64 -> 128
        # no pooling here (保持 H/4, W/4)

        self.flatten_dim = 128 * 6 * 6 
        self.fc = nn.Linear(self.flatten_dim, encoding_dim)
        self.encoding_dim = encoding_dim
        # 存储特征图的通道数及空间尺寸（不含 batch 维度）

    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return v: latent vector, dim: (Batch_size, encoding_dim)
        '''
        
        '''
        TODO: Implement the forward pass of the Encoder.

        Steps:
        1. Pass the input through the convolutional layers and max pooling.
        2. Flatten the output and pass it through the linear layer to obtain the latent vector.
        3. Return the latent vector.
        '''
        # Conv + ReLU + Pool
        x = F.relu(self.conv1(x))   # (B, 32, H, W)
        x = self.pool(x)            # (B, 32, H/2, W/2)

        x = F.relu(self.conv2(x))   # (B, 64, H/2, W/2)
        x = self.pool(x)            # (B, 64, H/4, W/4)

        x = F.relu(self.conv3(x))   # (B, 128, H/4, W/4)

        x = x.view(x.size(0), -1)   # Flatten
        v = self.fc(x)
        return v


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Decoder, self).__init__()
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        
        '''
        TODO: Implement the Decoder.

        Requirements:
        1. Use a linear layer to map the latent vector back to the feature map dimensions.
        2. Use transposed convolutional layers to upsample the feature maps.
        3. Ensure the output has the same dimensions as the input image.
        '''
        self.encoding_dim = encoding_dim

        self.flatten_dim = 128 * 6 * 6
        self.fc = nn.Linear(encoding_dim, self.flatten_dim)

        # 反卷积（上采样）
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        #self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # H/4 -> H/2
        #self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2)  # H/2 -> H
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) 
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, v):
        '''
        v: latent vector, dim: (Batch_size, encoding_dim)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        '''
        
        '''
        TODO: Implement the forward pass of the Decoder.

        Steps:
        1. Pass the latent vector through the linear layer to reconstruct the feature maps.
        2. Pass the feature maps through transposed convolutional layers to upsample them.
        3. Return the reconstructed images.
        '''
        x = self.fc(v)
        x = x.view(-1, 128, 6, 6)   # Reshape 回特征图形状

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x)) # 确保输出在 [0, 1]
        return x


# Combine the Encoder and Decoder to make the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(encoding_dim)
        self.decoder = Decoder(encoding_dim)

    def forward(self, x):
        '''
        TODO: Implement the forward pass of the Autoencoder.

        Steps:
        1. Pass the input through the Encoder to obtain the latent vector.
        2. Pass the latent vector through the Decoder to reconstruct the input.
        3. Return the reconstructed images.
        '''
        v = self.encoder(x)
        x_recon = self.decoder(v)
        return x_recon
    
    @property
    def name(self):
        return "AE"