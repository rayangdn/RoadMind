import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

def compute_ade_fde(pred_trajs, future, include_heading=False):    
    # Compute the L2 distance between predicted and ground truth trajectories
    if not include_heading:
        pred_trajs = pred_trajs[:, :, :2]
        future = future[:, :, :2]
        
    ade = torch.mean(torch.norm(pred_trajs - future, dim=2))
    fde = torch.mean(torch.norm(pred_trajs[:, -1] - future[:, -1], dim=1))
    
    return ade.item(), fde.item()

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
        x = self.pool(features) # [batch_size, 1280, 1, 1] 
        x = x.view(x.size(0), -1)  # [batch_size, 1280]
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
            dropout=dropout_rate
        )
        
    def forward(self, x):
        
        _, h_n = self.encoder(x)
        
        # Get the last hidden state
        x = h_n[-1]
        
        return x

class DepthDecoder(nn.Module):
    def __init__(self, in_channels=1280):
        super(DepthDecoder, self).__init__()
        
        # Upsampling blocks
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.up4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        # Final layer to produce depth map
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Normalize depth to [0,1]
        )
    
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)
        return x
    
class SemanticDecoder(nn.Module):
    def __init__(self, in_channels=1280, num_classes=15):  # 15 semantic classes 
        super(SemanticDecoder, self).__init__()
        
        # Upsampling blocks 
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.up4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        # Final layer to produce semantic map 
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)
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
        self.image_encoder = ImageEncoder(output_dim=image_embed_dim)
        
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

        # Process camera through  image encoder
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
    def __init__(self, hidden_dim=128, image_embed_dim=256, dropout_rate=0.3,
                 num_layers_gru=1, include_heading=False, include_dynamics=False,
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

        print("==================MODEL PARAMETERS==================")
        print(f"Hidden dim: {hidden_dim}, Image embed dim: {image_embed_dim}, Dropout rate: {dropout_rate}")
        print(f"Num Layers GRU: {num_layers_gru}, Include heading: {include_heading}, Include dynamics: {include_dynamics}")
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
        outputs = self(camera, history)
        
        # Compute loss
        loss = self.model.compute_loss(outputs, future)
        self.log('train_loss', loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        camera = batch['camera']
        history = batch['history']
        future = batch['future']

        # Forward pass
        outputs = self(camera, history)
        
        # Compute loss
        loss = self.model.compute_loss(outputs, future)
        
        # Compute ADE and FDE
        ade, fde = compute_ade_fde(outputs, future, self.include_heading)
        
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
            min_lr=1e-6,        
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