import torch
import torch.nn as nn

MLPAE_ENCODING_DIM = 64

class MLPAutoencoder(nn.Module):
    def __init__(
            self, 
            encoding_dim:int,
            img_width:int, 
            img_height:int, 
            img_channel:int=3
    ):
        '''
        Define the MLP autoencoder structure.
        
        Args:
            encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        super().__init__()
        
        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel
        self.input_dim = img_channel * img_width * img_height 

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim)   
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.input_dim),
            nn.Sigmoid()   
        )

    def forward(
            self, 
            x:torch.Tensor      #(Batch_size, IMG_CHANNEL, IMG_WIDTH, IMG_HEIGHT)
    ):
        '''
        The forward pass of the model
        Args:
            x: input images
        '''
        batch_size = x.size(0)

        # flatten
        x = x.view(batch_size, -1)

        # encode
        z = self.encoder(x)

        # decode
        x_recon = self.decoder(z)

        # reshape
        x_recon = x_recon.view(batch_size, self.img_channel, self.img_width, self.img_height)

        return x_recon
    
    @property
    def name(self):
        return "MLPAE"
