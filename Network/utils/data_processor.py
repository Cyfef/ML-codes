import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid


def create_flower_dataloaders(
        batch_size:int, 
        root:str, 
        IMG_WIDTH:int, 
        IMG_HEIGHT:int, 
        num_worker:int=0
    ):
    '''
    Create training and validation dataloaders for a flower image dataset

    Args:
        batch_size: Number of samples per batch
        root: Root directory path of the dataset
        IMG_WIDTH: Target image width after resizing.
        IMG_HEIGHT: Target image height after resizing.
        num_worker: Number of subprocesses used for data loading
    '''

    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)  # ensure 3 channels
    ])

    flower_dataset = datasets.ImageFolder(root=root, transform=transform)

    train_size = int(0.8 * len(flower_dataset))     #train 0.8
    valid_size = len(flower_dataset) - train_size   #valid 0.2

    train_dataset, valid_dataset = random_split(flower_dataset, [train_size, valid_size])

    training_dataloaders = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_worker, 
        pin_memory=True     # speed up GPU transfer
    )
    validation_dataloaders = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_worker, 
        pin_memory=True
    )
    
    return training_dataloaders, validation_dataloaders


def show_images(
        images:torch.Tensor,    #(B, C, H, W) 
        nmax:int=64
    ):
    '''
    Visualize a batch of images in a grid

    Args:
        images: Input image batch
        max: Maximum number of images to display
    '''
    img_grid = make_grid(images[:nmax], nrow=8).permute(1, 2, 0)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_grid)
    plt.axis('off')
    plt.show()


def show_recover_results(
        images:torch.Tensor,            #(B, C, H, W)
        recover_images:torch.Tensor,    #(B, C, H, W)
        save_path:str
    ):
    '''
    Show images on two rows 

    Args:
        images: Original images
        recover_images: Reconstructed images
        save_path: Path to save the visualization image
    '''
    
    num_images = images.shape[0]
    
    # show on two rows, first row is the original images, second row is the recovered images
    img_grid = make_grid(torch.cat([images, recover_images]), nrow=num_images).permute(1, 2, 0)
    plt.figure(figsize=(20, 20))
    
    img_grid = img_grid.cpu().numpy()
    
    plt.imshow(img_grid)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=1000)
        