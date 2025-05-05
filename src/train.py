import os
import torch
import torch.nn as nn

from utils import save_checkpoint, load_checkpoint
    
def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=50, patience=30, model_dir='../model', output_dir='../output/checkpoints', device='cuda'):
    
    start_epoch = 0
    saved_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path=os.path.join(output_dir, 'checkpoint.pth'))
    if saved_epoch > 0:
        start_epoch = saved_epoch + 1
        print(f"Resuming training from epoch {start_epoch}")

    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
    val_ade = []
    val_fde = []
    
    best_val_loss = float('inf')
    counter = 0
    
    for epoch in range(start_epoch, epochs):
        
        # Training 
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            camera = batch['camera'].to(device)
            sdc_history = batch['sdc_history_feature'].to(device)
            sdc_future = batch['sdc_future_feature'].to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(camera, sdc_history)
            loss = criterion(outputs, sdc_future)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        ade_total = 0.0
        fde_total =0.0
        
        with torch.no_grad():
            for batch in val_loader:
                camera = batch['camera'].to(device)
                sdc_history = batch['sdc_history_feature'].to(device)
                sdc_future = batch['sdc_future_feature'].to(device)
                
                outputs = model(camera, sdc_history)
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
        
        scheduler.step(avg_train_loss)
        
        print(f'Epoch {epoch+1}/{epochs} , Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, ADE: {avg_ade:.4f}, FDE: {avg_fde:.4f}, Learning rate: {scheduler.get_last_lr()}')
         
        save_checkpoint(model, optimizer, epoch, avg_train_loss, 
                        checkpoint_path=os.path.join(output_dir, 'checkpoint.pth'),
                        store_checkpoint_for_every_epoch=False)
                
         # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            
            # Save only the model state dictionary
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
            print(f"Best Model saved at epoch {epoch+1}/{epochs} with validation loss: {avg_val_loss:.4f}")
            
        else:
            counter += 1
            print(f"Early stopping counter: {counter}/{patience}")
            
        # Check early stopping condition
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return train_losses, val_losses, val_ade, val_fde
