import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

def compute_ade_fde(pred_trajs, future, include_heading=False):
        
    # Compute the L2 distance between predicted and ground truth trajectories
    pred_trajs = pred_trajs[:, :, :2]
    future = future[:, :, :2]
        
    ade = torch.mean(torch.norm(pred_trajs - future, dim=2))
    fde = torch.mean(torch.norm(pred_trajs[:, -1] - future[:, -1], dim=1))
    
    return ade.item(), fde.item()

# More weights for the last points
def weighted_mse_loss(prediction, target, weight_schedule='linear', max_weight=5000.0, eps=1e-6):
    batch_size, seq_len, dims = prediction.shape
    
    # Generate weights based on schedule
    t = torch.linspace(1.0, max_weight, steps=seq_len, device=prediction.device)
    
    if weight_schedule == 'quadratic':
        # Smoother increase with mid-points getting moderate weights
        alpha = torch.sqrt(torch.linspace(1.0, max_weight**2, steps=seq_len, device=prediction.device))
    elif weight_schedule == 'exp':
        # More aggressive weighting of later points
        alpha = torch.exp(torch.linspace(0, np.log(max_weight), steps=seq_len, device=prediction.device))
    else:  # default to linear
        alpha = t
    
    # Normalize weights to avoid changing overall loss magnitude too much
    alpha = alpha / alpha.mean()
    
    weights = alpha.view(1, seq_len, 1).expand(batch_size, seq_len, dims)
    
    squared_error = weights * (prediction - target)**2
    
    # Mean over all dimensions
    return squared_error.mean()

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super(ImageEncoder, self).__init__()
        
        self.output_dim = output_dim
        
        # Use a pretrained EfficientNet model
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Remove the classifier and pooling layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2]) 
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Pool to (batch, channels, 1, 1)
        
        self.fc = nn.Linear(1280, self.output_dim)  # EfficientNet last conv outputs 1280 channels

    def forward(self, x):
        
        # Pass through EfficientNet backbone
        features = self.backbone(x)
        
        # Global average pooling
        x = self.pool(features)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)

        return x

class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=1, dropout_rate=0.3):
        super(TrajectoryEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        
    def forward(self, x):
        
        _, h_n = self.encoder(x)
        
        # Get the last hidden state
        x = h_n[-1]
        
        return x
        
class TrajectoryDecoder(nn.Module):
    def __init__(self, input_dim=256, output_dim=120):
        super(TrajectoryDecoder, self).__init__()
        
        self.decoder = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        
        # Pass through decoder
        output = self.decoder(x)
        
        return output

class RoadMind(nn.Module):
    def __init__(self, input_hist_dim=3, hidden_dim=128, image_embed_dim=256, num_layers_gru=1,
                 output_seq_len=60, dropout_rate=0.3, include_heading=False):
        super(RoadMind, self).__init__()        
        
        self.input_hist_dim = input_hist_dim
        self.hidden_dim = hidden_dim
        self.output_hist_dim = hidden_dim
        self.image_embed_dim = image_embed_dim
        self.num_layers_gru = num_layers_gru
        self.output_seq_len = output_seq_len
        self.dropout_rate = dropout_rate
        self.include_heading = include_heading
        self.output_futur_dim = 2 if not include_heading else 3
        
        # Image encoder
        self.image_encoder = ImageEncoder(
            output_dim=image_embed_dim
        )
        
        # Trajectory encoder
        self.trajectory_encoder = TrajectoryEncoder(
            input_dim=input_hist_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers_gru,
            dropout_rate=dropout_rate
        )

        # Fusion layer
        self.fusion_output_dim = 4*hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(image_embed_dim + self.output_hist_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4*hidden_dim, self.fusion_output_dim),
            nn.ReLU(),
        )
        
        self.trajectory_decoder = TrajectoryDecoder(
            input_dim=self.fusion_output_dim, 
            output_dim=self.output_seq_len * self.output_futur_dim,
        )
        
    def forward(self, camera, history):

        # Process camera images through image encoder
        image_features = self.image_encoder(camera)
        
        # Process history through trajectory encoder
        history_hidden = self.trajectory_encoder(history)
        
        # Concatenate image features and trajectory hidden state
        combined_features = torch.cat((image_features, history_hidden), dim=1)

        # Pass through fusion layer
        fusion_output = self.fusion(combined_features)
        
        # Pass through trajectory decoder
        output = self.trajectory_decoder(fusion_output)
        output = output.view(-1, self.output_seq_len, self.output_futur_dim)
            
        return output
         
    def compute_loss(self, traj_pred, future):
        
        if not self.include_heading:
            future = future[:, :, :2]

        loss = nn.MSELoss()(traj_pred, future)
        
        return loss
        
class LightningRoadMind(pl.LightningModule):
    def __init__(self, hidden_dim=128, image_embed_dim=256, num_layers_gru=1, 
                 dropout_rate=0.3, weight_depth=10, weight_semantic=0.2, 
                 include_heading=False, include_dynamics=False,
                 use_depth_aux=False, use_semantic_aux=False,
                 lr=1e-4, weight_decay=1e-5, scheduler_factor=0.5, 
                 scheduler_patience=5):
        
        super(LightningRoadMind, self).__init__()
            
        # Initialize the model
        self.model = RoadMind(
            input_hist_dim=8 if include_dynamics else 3,
            hidden_dim=hidden_dim,
            image_embed_dim=image_embed_dim,
            num_layers_gru=num_layers_gru,
            dropout_rate=dropout_rate,
            include_heading=include_heading
        )
        
        # Optimization parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.include_heading = include_heading

        print("\n==================MODEL PARAMETERS==================")
        print(f"Hidden dim: {hidden_dim}, Image embed dim: {image_embed_dim}, Num Layers GRU: {num_layers_gru}")
        print(f"Dropout rate: {dropout_rate}, Include heading: {include_heading}, Include dynamics: {include_dynamics}")
        print("=======================================================\n")
        
        print("==================OPTIMIZATION PARAMETERS==================")
        print(f"Learning rate: {lr}, Weight decay: {weight_decay}")
        print(f"Scheduler factor: {scheduler_factor}, Scheduler patience: {scheduler_patience}")
        print("=======================================================\n")
        
        self.save_hyperparameters()
    
    def forward(self, camera, history):
        return self.model(camera, history)
    
    def training_step(self, batch, batch_idx):
        
        camera = batch['camera']           
        history = batch['history']        
        future = batch['future'] 
             
        # Forward pass
        traj_pred = self(camera, history)
        
        # Compute loss
        loss = self.model.compute_loss(traj_pred, future)
        
        # Log loss
        self.log('train_loss', loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        camera = batch['camera']
        history = batch['history']
        future = batch['future']

        # Forward pass
        traj_pred = self(camera, history)
        
        # Compute losses
        loss = self.model.compute_loss(traj_pred, future)
        
        # Compute ADE and FDE
        ade, fde = compute_ade_fde(traj_pred, future)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_ade', ade, prog_bar=True, sync_dist=True)
        self.log('val_fde', fde, prog_bar=True, sync_dist=True)

        return {'val_loss': loss, 'val_ade': ade, 'val_fde': fde}

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',          
            factor=self.scheduler_factor,          
            patience=self.scheduler_patience,          
            min_lr=1e-7,        
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_ade',
                'interval': 'epoch',
                'frequency': 1,
                'reduce_on_plateau': True  
            }
        }