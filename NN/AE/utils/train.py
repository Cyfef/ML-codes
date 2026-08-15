import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Callable


def train(
        optimizer: Optimizer, 
        scheduler: Callable[[int], float], 
        model: torch.nn.Module, 
        training_dataloader: DataLoader, 
        validation_dataloader: DataLoader,
        num_epochs: int,
        early_stopping_patience: int,
        device: torch.device,
        model_save_root: str,
        loss_fn: Callable = F.mse_loss
    )->None:
    '''
    Train an AutoEncoder model with validation, learning rate scheduling, model checkpointing, and early stopping

    Args:
        optimizer: PyTorch optimizer used to update model parameters
        scheduler: Learning rate scheduler 
        model: The AutoEncoder model to be trained
        training_dataloader: DataLoader for training dataset
        validation_dataloader: DataLoader for validation dataset
        num_epochs: Total number of training epochs
        early_stopping_patience: Number of validation checks with no improvement before stopping training early
        device: Device to run training on
        model_save_root: Directory path to save the best model
        loss_fn: Loss function to measure reconstruction error
    '''

    model.train()
    
    #name
    model_name = model.name
    save_model_name = f"Best_{model_name}.pth"
    os.makedirs(model_save_root, exist_ok=True)
    
    min_valid_loss = float('inf')
    # Training Loop
    avg_train_loss = 10000.
    avg_valid_loss = 10000.
    no_improve = 0      # early stopping sign
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        # Adjust the learning rate
        lr = scheduler(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # Training all batches
        for images, _ in training_dataloader:
            images = images.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        avg_train_loss = sum(train_losses) / len(train_losses)


        # Validation every 3 epochs
        if epoch % 3 == 0:
            model.eval()
            valid_losses = []
            with torch.no_grad():
                for images, _ in validation_dataloader:
                    images = images.to(device)
                    
                    outputs = model(images)
                    loss = loss_fn(outputs, images)
                    
                    valid_losses.append(loss.item())
                avg_valid_loss = sum(valid_losses) / len(valid_losses)
            
            if avg_valid_loss < min_valid_loss:
                min_valid_loss = avg_valid_loss
                no_improve = 0

                save_path = os.path.join(model_save_root, save_model_name)
                torch.save(model.state_dict(), save_path)
                print(f"Model saved at epoch {epoch}, val_loss={avg_valid_loss:.6f}")
            else:
                no_improve += 1

            # early stopping
            if no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
        print(f"Epoch: {epoch}, Train Loss: {avg_train_loss}, Valid Loss: {avg_valid_loss}")
