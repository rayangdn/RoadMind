import os
import torch
import torch.nn as nn
from utils import save_checkpoint

def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=10, start_epoch=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
    val_ade = []
    val_fde = []
    
    for epoch in range(start_epoch, epochs):
        # Training 
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            camera = batch['camera'].to(device)
            driving_command = batch['driving_command'].to(device)
            sdc_history = batch['sdc_history_feature'].to(device)
            sdc_future = batch['sdc_future_feature'].to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(camera, driving_command, sdc_history)
            loss = criterion(outputs, sdc_future)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Save checkpoint
        checkpoints_dir = "../outputs/checkpoints"
        os.makedirs(checkpoints_dir, exist_ok=True)
        if ((epoch + 1) % 5 == 0 or (epoch + 1) == epochs):
            save_checkpoint(model, optimizer, epoch, avg_train_loss, os.path.join(checkpoints_dir, "checkpoint.pth"))
                           
        # Validation
        model.eval()
        val_loss = 0.0
        ade_total = 0.0
        fde_total =0.0
        
        with torch.no_grad():
            for batch in val_loader:
                camera = batch['camera'].to(device)
                driving_command = batch['driving_command'].to(device)
                sdc_history = batch['sdc_history_feature'].to(device)
                sdc_future = batch['sdc_future_feature'].to(device)
                
                outputs = model(camera, driving_command, sdc_history)
                loss = criterion(outputs, sdc_future)
                val_loss += loss.item()
                
                ade = torch.norm(outputs[:, :, :2] - sdc_future[:, :, :2], p=2, dim=-1).mean()
                fde = torch.norm(outputs[:, -1, :2] - sdc_future[:, -1, :2], p=2, dim=-1).mean()
                ade_total += ade
                fde_total += fde
        
        avg_val_loss = val_loss / len(val_loader)
        avg_ade = ade_total / len(val_loader)
        avg_fde = fde_total / len(val_loader)
        val_losses.append(avg_val_loss)
        val_ade.append(avg_ade.detach().cpu().numpy())
        val_fde.append(avg_fde.detach().cpu().numpy())
        
        scheduler.step(avg_val_loss)
        print("Learning rate:", scheduler.get_last_lr())
         
        print(f'Epoch {epoch+1}/{epochs} , Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, ADE: {avg_ade:.4f}, FDE: {avg_fde:.4f}')
    
    return train_losses, val_losses, val_ade, val_fde