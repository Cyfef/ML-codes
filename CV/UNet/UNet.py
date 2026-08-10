import torch
import torch.nn as nn
from wandb_utils import *

class Double_Conv_block(nn.Module):
    def __init__(
            self,
            in_channels:int,
            out_channels:int
        ):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,
                x:torch.Tensor):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self,
                 num_class:int):
        super().__init__()

        self.conv_down1=Double_Conv_block(3,64)
        self.conv_down2=Double_Conv_block(64,128)
        self.conv_down3=Double_Conv_block(128,256)
        self.conv_down4=Double_Conv_block(256,256)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 

        self.conv_up1=Double_Conv_block(256+256,128)
        self.conv_up2=Double_Conv_block(128+128,64)
        self.conv_up3=Double_Conv_block(64+64,64)

        self.conv_last=nn.Conv2d(64,num_class,kernel_size=1)

        self.num_class=num_class

    def forward(
            self,
            x:torch.Tensor
        ):

        conv1=self.conv_down1(x)
        x=self.maxpool(conv1)

        conv2=self.conv_down2(x)
        x=self.maxpool(conv2)

        conv3=self.conv_down3(x)
        x=self.maxpool(conv3)

        x=self.conv_down4(x)

        x=self.upsample(x)
        x=torch.cat([x,conv3],dim=1)
        x=self.conv_up1(x)

        x=self.upsample(x)
        x=torch.cat([x,conv2],dim=1)
        x=self.conv_up2(x)

        x=self.upsample(x)
        x=torch.cat([x,conv1],dim=1)
        x=self.conv_up3(x)

        out=self.conv_last(x)
        return out
    

class UNetTrainer():
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
              log_interval:int=50):

        wandb_init()
        
        self.model.train()
        iter_count=0
    
        for epoch in range(1,num_epochs+1):
            for batch in train_dataloader:
                imgs = batch['image'].to(self.device, dtype=torch.float32)   
                labels = batch['mask'].to(self.device, dtype=torch.long)     
    
                self.optimizer.zero_grad()
    
                logits=self.model(imgs)
                loss=torch.nn.functional.cross_entropy(logits,labels)
    
                loss.backward()
                self.optimizer.step()

                wandb_log({
                    "train/loss": loss,
                    "epoch": epoch,
                    })
    
                if iter_count % log_interval == 0:
                    print(f'Iter: {iter_count}, Loss: {loss.item():.4}')
    
                iter_count += 1

        wandb_finish()