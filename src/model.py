import torch
import torch.nn as nn
import torch.nn.functional as F


### DO NOT MODIFY THIS MODEL ###
class SpatialAttention(nn.Module):
    """
    Spatial attention module that helps the model focus on relevant parts of the image
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        
    def forward(self, x):
        # First convolution to reduce channels
        attn = self.conv1(x)
        attn = F.relu(attn)
        
        # Second convolution to generate attention map
        attn = self.conv2(attn)
        
        # Apply sigmoid to get attention weights between 0 and 1
        attn = torch.sigmoid(attn)
        
        # Apply attention weights to input feature map
        return x * attn.expand_as(x)

class TemporalAttention(nn.Module):
    """
    Temporal attention module that helps the model focus on relevant timesteps
    """
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)
        
    def forward(self, gru_outputs):
        # gru_outputs shape: [batch_size, seq_len, hidden_size]
        
        # Calculate attention scores
        attn_scores = self.attn(gru_outputs)  # [batch_size, seq_len, 1]
        attn_scores = attn_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]
        
        # Apply attention weights to get context vector
        context = torch.bmm(attn_weights, gru_outputs)  # [batch_size, 1, hidden_size]
        context = context.squeeze(1)  # [batch_size, hidden_size]
        
        return context

class CrossAttention(nn.Module):
    """
    Cross-attention module that helps relate visual features to motion features
    """
    def __init__(self, query_dim, key_dim, value_dim, attn_dim=64):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(query_dim, attn_dim)
        self.key_proj = nn.Linear(key_dim, attn_dim)
        self.value_proj = nn.Linear(value_dim, value_dim)  # Usually we project values to same dim
        self.scale = torch.sqrt(torch.tensor(attn_dim, dtype=torch.float32))
        
    def forward(self, query, key, value):
        # Project query, key, value
        query_proj = self.query_proj(query)  # [batch_size, query_dim] -> [batch_size, attn_dim]
        key_proj = self.key_proj(key)        # [batch_size, key_dim] -> [batch_size, attn_dim]
        value_proj = self.value_proj(value)  # [batch_size, value_dim] -> [batch_size, value_dim]
        
        # Calculate attention scores
        attn_scores = torch.matmul(query_proj.unsqueeze(1), key_proj.unsqueeze(2)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=2)
        
        # Apply attention weights to get context vector
        context = attn_weights * value_proj.unsqueeze(1)
        
        return context.squeeze(1)

class RoadMind(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(RoadMind, self).__init__()
        
        # Increased channel capacity in convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        # Spatial attention after last conv layer
        self.spatial_attention = SpatialAttention(128)
        
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        
        # Calculate flattened size
        self.flatten_size = 128 * 5 * 5
        
        # Using GRU
        self.gru = nn.GRU(
            input_size=3, 
            hidden_size=64,
            num_layers=2,
            batch_first=True, 
            dropout=dropout_rate if 2 > 1 else 0
        )
        
        # Temporal attention for GRU outputs
        self.temporal_attention = TemporalAttention(64)
        
        self.dropout_gru = nn.Dropout(dropout_rate)
        
        # Cross-attention between visual and motion features
        self.cross_attention = CrossAttention(
            query_dim=64,    # GRU hidden size
            key_dim=self.flatten_size,  # CNN feature size
            value_dim=self.flatten_size  # CNN feature size
        )
        
        # Fusion and prediction with increased capacity
        self.fusion = nn.Sequential(
            nn.Linear(self.flatten_size + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate/4),
            
            nn.Linear(128, 60 * 3)  # Predict 60 timesteps, each with x, y, heading
        )

        # Transformer encoder layer for comined features
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.flatten_size + 64,
            nhead=8
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=1
        )
        
        # Weight initialization
        self._initialize_weights()
           
    def forward(self, image, motion_history):
        # Process image (batch_size, 3, 200, 300)
        x_img = F.relu(self.bn1(self.conv1(image)))
        x_img = self.dropout1(x_img)
        
        x_img = F.relu(self.bn2(self.conv2(x_img)))
        x_img = self.dropout2(x_img)
        
        x_img = F.relu(self.bn3(self.conv3(x_img)))
        x_img = self.dropout3(x_img)
        
        x_img = F.relu(self.bn4(self.conv4(x_img)))
        x_img = self.dropout4(x_img)
        
        # Apply spatial attention
        x_img = self.spatial_attention(x_img)
        
        x_img = self.pool(x_img)
        x_img_flat = x_img.reshape(-1, self.flatten_size)
        
        # Process motion history (batch_size, 21, 3) with GRU
        gru_outputs, hidden = self.gru(motion_history)
        
        # Apply temporal attention to GRU outputs
        x_mot = self.temporal_attention(gru_outputs)
        x_mot = self.dropout_gru(x_mot)
        
        # Apply cross-attention between motion and visual features
        # Using motion as query and visual as key/value
        attended_visual = self.cross_attention(x_mot, x_img_flat, x_img_flat)
        
        # Concatenate features - using the attended visual features
        combined = torch.cat([attended_visual, x_mot], dim=-1)
        
        # For transformer
        attended_visual = self.cross_attention(x_mot, x_img_flat, x_img_flat)
        combined = torch.cat([attended_visual, x_mot], dim=-1)

        combined = self.transformer_encoder(combined.unsqueeze(1)).squeeze(1)

        # Generate trajectory
        trajectory = self.fusion(combined)
        trajectory = trajectory.reshape(-1, 60, 3)  # Reshape to (batch_size, 60, 3)
        
        return trajectory
    
    def _initialize_weights(self):
        """Initialize model weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)