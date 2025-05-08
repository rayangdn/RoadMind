import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import save_checkpoint, load_checkpoint
    
def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=50, patience=30, 
                model_dir='../model', output_dir='../output/checkpoints', device='cuda',
                use_depth_aux=False, use_semantic_aux=False, 
                lambda_depth=0.1, lambda_semantic=0.1):
    """
    Train the model with optional auxiliary tasks
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for weight updates
        scheduler: Learning rate scheduler
        epochs: Maximum number of epochs to train
        patience: Patience for early stopping
        model_dir: Directory to save best model
        output_dir: Directory to save checkpoints
        device: Device to use for training
        use_depth_aux: Whether to use depth estimation auxiliary task
        use_semantic_aux: Whether to use semantic segmentation auxiliary task
        lambda_depth: Weight for depth estimation loss
        lambda_semantic: Weight for semantic segmentation loss
    """
    
    start_epoch = 0
    saved_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path=os.path.join(output_dir, 'checkpoint.pth'))
    if saved_epoch > 0:
        start_epoch = saved_epoch + 1
        print(f"Resuming training from epoch {start_epoch}")

    # Criteria for different tasks
    traj_criterion = nn.MSELoss()
    depth_criterion = nn.L1Loss()  # L1 loss is typically better for depth
    semantic_criterion = nn.CrossEntropyLoss()  # For semantic segmentation
    
    train_losses = []
    train_traj_losses = []
    train_depth_losses = []
    train_semantic_losses = []
    val_losses = []
    val_traj_losses = []
    val_depth_losses = []
    val_semantic_losses = []
    val_ade = []
    val_fde = []
    
    best_val_loss = float('inf')
    counter = 0
    
    for epoch in range(start_epoch, epochs):
        
        # Training 
        model.train()
        train_loss = 0.0
        train_traj_loss = 0.0
        train_depth_loss = 0.0
        train_semantic_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            camera = batch['camera'].to(device)
            sdc_history = batch['sdc_history_feature'].to(device)
            sdc_future = batch['sdc_future_feature'].to(device)
            
            # Get additional inputs for auxiliary tasks
            depth = batch['depth'].to(device) if use_depth_aux else None
            semantic = batch['semantic_label'].to(device) if use_semantic_aux else None
            
            optimizer.zero_grad()
            
            # Forward pass - now returns multiple outputs
            outputs = model(camera, sdc_history)

            # Unpack outputs based on which auxiliary tasks are enabled
            if use_depth_aux and use_semantic_aux:
                traj_output, depth_output, semantic_output = outputs
            elif use_depth_aux:
                traj_output, depth_output, _ = outputs
            elif use_semantic_aux:
                traj_output, _, semantic_output = outputs
            else:
                traj_output, _, _ = outputs
            print(depth.shape)
            print(semantic.shape)
            print(depth_output.shape)
            print(semantic_output.shape)
            # Calculate trajectory loss
            t_loss = traj_criterion(traj_output, sdc_future)
            train_traj_loss += t_loss.item()
            
            # Initialize total loss with trajectory loss
            loss = t_loss
            
            # Add depth loss if enabled
            if use_depth_aux and depth is not None:
                d_loss = depth_criterion(depth_output, depth)
                train_depth_loss += d_loss.item()
                loss += lambda_depth * d_loss
            
            # Add semantic loss if enabled
            if use_semantic_aux and semantic is not None:
                # Reshape for CrossEntropyLoss: [B, C, H, W]
                s_loss = semantic_criterion(semantic_output, semantic)
                train_semantic_loss += s_loss.item()
                loss += lambda_semantic * s_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_train_traj_loss = train_traj_loss / len(train_loader)
        avg_train_depth_loss = train_depth_loss / len(train_loader) if use_depth_aux else 0
        avg_train_semantic_loss = train_semantic_loss / len(train_loader) if use_semantic_aux else 0
        
        train_losses.append(avg_train_loss)
        train_traj_losses.append(avg_train_traj_loss)
        train_depth_losses.append(avg_train_depth_loss)
        train_semantic_losses.append(avg_train_semantic_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_traj_loss = 0.0
        val_depth_loss = 0.0
        val_semantic_loss = 0.0
        ade_total = 0.0
        fde_total = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                camera = batch['camera'].to(device)
                sdc_history = batch['sdc_history_feature'].to(device)
                sdc_future = batch['sdc_future_feature'].to(device)
                
                # Get additional inputs for auxiliary tasks
                depth = batch['depth'].to(device) if use_depth_aux else None
                semantic = batch['semantic_label'].to(device) if use_semantic_aux else None
                
                # Forward pass
                outputs = model(camera, sdc_history)
                
                # Unpack outputs
                if use_depth_aux and use_semantic_aux:
                    traj_output, depth_output, semantic_output = outputs
                elif use_depth_aux:
                    traj_output, depth_output, _ = outputs
                elif use_semantic_aux:
                    traj_output, _, semantic_output = outputs
                else:
                    traj_output, _, _ = outputs

                # Calculate trajectory loss and metrics
                t_loss = traj_criterion(traj_output, sdc_future)
                val_traj_loss += t_loss.item()
                
                # Initialize total validation loss
                v_loss = t_loss
                
                # Add depth loss if enabled
                if use_depth_aux and depth is not None:
                    d_loss = depth_criterion(depth_output, depth)
                    val_depth_loss += d_loss.item()
                    v_loss += lambda_depth * d_loss
                
                # Add semantic loss if enabled
                if use_semantic_aux and semantic is not None:
                    s_loss = semantic_criterion(semantic_output, semantic)
                    val_semantic_loss += s_loss.item()
                    loss += lambda_semantic * s_loss
                
                val_loss += v_loss.item()
                
                # Calculate trajectory metrics
                ade = torch.norm(traj_output[:, :, :2] - sdc_future[:, :, :2], p=2, dim=-1).mean()
                fde = torch.norm(traj_output[:, -1, :2] - sdc_future[:, -1, :2], p=2, dim=-1).mean()
                ade_total += ade
                fde_total += fde
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_traj_loss = val_traj_loss / len(val_loader)
        avg_val_depth_loss = val_depth_loss / len(val_loader) if use_depth_aux else 0
        avg_val_semantic_loss = val_semantic_loss / len(val_loader) if use_semantic_aux else 0
        avg_ade = ade_total / len(val_loader)
        avg_fde = fde_total / len(val_loader)
        
        # Record metrics for plotting
        val_losses.append(avg_val_loss)
        val_traj_losses.append(avg_val_traj_loss)
        val_depth_losses.append(avg_val_depth_loss)
        val_semantic_losses.append(avg_val_semantic_loss)
        val_ade.append(avg_ade.detach().cpu().numpy())
        val_fde.append(avg_fde.detach().cpu().numpy())
        
        # Update learning rate scheduler
        scheduler.step(avg_train_loss)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train - Loss: {avg_train_loss:.4f}, Traj: {avg_train_traj_loss:.4f}, '
              f'Depth: {avg_train_depth_loss:.4f}, Semantic: {avg_train_semantic_loss:.4f}')
        print(f'  Val   - Loss: {avg_val_loss:.4f}, Traj: {avg_val_traj_loss:.4f}, '
              f'Depth: {avg_val_depth_loss:.4f}, Semantic: {avg_val_semantic_loss:.4f},'
              f'ADE: {avg_ade:.4f}, FDE: {avg_fde:.4f}')
        
        print(f'  Learning rate: {scheduler.get_last_lr()}')
         
        # Save checkpoint
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
    
    return train_losses, train_traj_losses, train_depth_losses, train_semantic_losses, val_losses, val_traj_losses, val_depth_losses, val_semantic_losses, val_ade, val_fde,

