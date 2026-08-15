import torch
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"

from wandb_utils import *


class PointNetfeat(nn.Module):
    '''
        The feature extractor in PointNet, corresponding to the left MLP in the pipeline figure.
        Args:
            d: the dimension of the global feature, default is 1024.
            segmentation: whether to perform segmentation, default is True.
    '''
    def __init__(self, 
                 segmentation = True, 
                 d=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, d, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(d)

        self.maxpool=nn.AdaptiveMaxPool1d(1)

        self.segmentation=segmentation
        self.d = d

    def forward(self, x):
        '''
            If segmentation == True
                return the concatenated global feature and local feature. # (B, d+64, N)
            If segmentation == False
                return the global feature, and the per point feature for cruciality visualization in question b). # (B, d), (B, N, d)
            Here, B is the batch size, N is the number of points, d is the dimension of the global feature.
        '''

        # x input shape: (B, N, 3)
        x = x.transpose(1, 2)   # (B, 3, N)

        x1 = F.relu(self.bn1(self.conv1(x)))    # (B, 64, N)
        x2 = F.relu(self.bn2(self.conv2(x1)))   # (B, 128, N)
        x3 = F.relu(self.bn3(self.conv3(x2)))   # (B, d, N)

        # global feature (B, d, 1)
        global_feat = self.maxpool(x3)

        if self.segmentation:
            global_feat_expanded = global_feat.repeat(1, 1, x.shape[2])
            return torch.cat([x1, global_feat_expanded], 1)     # (B, 64 + d, N)
        else:
            return global_feat.squeeze(-1), x3.transpose(1, 2) # (B, d), (B, N, d)


class PointNetCls(nn.Module):
    '''
        The classifier in PointNet, corresponding to the middle right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''
    def __init__(self, k=2):
        super().__init__()
        self.feat_extract=PointNetfeat(segmentation = False ,d=1024)
        self.fc1=nn.Linear(1024,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,k)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.num_classes=k

    def forward(self, x):
        '''
            return the log softmax of the classification result and the per point feature for cruciality visualization in question b). # (B, k), (B, N, d=1024)
        '''
        feat,output2=self.feat_extract(x)   # (B, d=1024), (B, N, d)
        
        x = F.relu(self.bn1(self.fc1(feat)))      # (B,512)
        x = F.relu(self.bn2(self.fc2(x)))         # (B,256)
             
        x=self.fc3(x)        # (B,k)
        
        result=F.log_softmax(x,dim=-1)    
        
        return result, output2
        
class PointNetSeg(nn.Module):
    '''
        The segmentation head in PointNet, corresponding to the lower right MLP in the pipeline figure.
        Args:
            k: the number of classes, default is 2.
    '''
    def __init__(self, k = 2):
        super().__init__()

        self.feat_extract=PointNetfeat(segmentation = True,d=1024)
        self.conv1=nn.Conv1d(64+1024,512,1)
        self.conv2=nn.Conv1d(512,256,1)
        self.conv3=nn.Conv1d(256,128,1)
        self.conv4=nn.Conv1d(128,k,1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.num_classes=k

    def forward(self, x):
        '''
            Input:
                x: the input point cloud. # (B, N, 3)
            Output:
                the log softmax of the segmentation result. # (B, N, k)
        '''
        feat=self.feat_extract(x)   # (B, 64 + d, N)

        x = F.relu(self.bn1(self.conv1(feat)))      # (B,512,N)
        x = F.relu(self.bn2(self.conv2(x)))         # (B,256,N)
        x = F.relu(self.bn3(self.conv3(x)))         # (B,128,N)
     
        x=self.conv4(x)        # (B,k,N)

        x=x.permute(0,2,1)  # (B,N,k)
        result=F.log_softmax(x,dim=-1)    

        return result


class PointNetClsTrainer():
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 dtype,
                 device):
                
        self.model=model.to(device)
        self.optimizer=optimizer
        self.scheduler=scheduler
        
        self.dtype=dtype
        self.device = device

    def train(self,
              num_epochs:int,
              train_dataloader,
              log_interval:int=10):
        wandb_init()
        self.model.train()

        iter_count=0
        for epoch in range(num_epochs):
            for data in train_dataloader:
                points, target = data
                points=points.to(self.device)
                target=target.to(self.device)
                target = target[:, 0]

                self.optimizer.zero_grad()

                pred, _ = self.model(points)

                loss = F.nll_loss(pred, target)
                loss.backward()
                self.optimizer.step()

                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().float().mean()

                wandb_log({
                            "train/loss": loss,
                            "train/accuracy": correct,
                            "train/iteration": iter_count,
                })
                
                if iter_count % log_interval == 0:
                    print(f'[{epoch}: {iter_count}] train loss: {loss.item()}, accuracy: {correct.item()}')
        
                iter_count +=1
        
            self.scheduler.step()
        wandb_finish()
     

class PointNetSegTrainer():
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 dtype,
                 device):
                
        self.model=model.to(device)
        self.optimizer=optimizer
        self.scheduler=scheduler
        
        self.dtype=dtype
        self.device = device

    def train(self,
              num_epochs:int,
              train_dataloader,
              log_interval:int=10):
        wandb_init()
        self.model.train()

        iter_count=0
        for epoch in range(num_epochs):
            for data in train_dataloader:
                points, target = data
                points=points.to(self.device)
                target=target.to(self.device)

                self.optimizer.zero_grad()

                pred = self.model(points)
                pred = pred.view(-1, self.model.num_classes)
                target = target.view(-1, 1)[:, 0] - 1

                loss = F.nll_loss(pred, target)
                loss.backward()
                self.optimizer.step()

                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().float().mean()

                if iter_count % log_interval==0:
                    print(f'[{epoch}: {iter_count}] train loss: {loss.item()}, accuracy: {correct.item()}')
                
                wandb_log({
                            "train/loss": loss,
                            "train/accuracy": correct,
                            "train/iteration": iter_count,
                })
                
                iter_count +=1
        
            self.scheduler.step()

        wandb_finish()