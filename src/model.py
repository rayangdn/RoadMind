import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class RoadMind(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(RoadMind, self).__init__()
        
        # ResNet18 backbone for processing camera input
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove the final fully connected layer
        self.visual_backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze the first 6 components
        ct = 0
        for child in self.visual_backbone.children():
            ct += 1
            if ct <= 7:
                for param in child.parameters():
                    param.requires_grad = False
        # Global pooling to get a fixed-size feature vector
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimension after pooling
        self.visual_feature_dim = 512
        
        # GRU for processing motion history
        self.motion_gru = nn.GRU(
            input_size=3,  # x, y, heading
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate if 2 > 1 else 0
        )
        
        # Fusion layer to combine visual and motion features
        self.fusion = nn.Sequential(
            nn.Linear(self.visual_feature_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Trajectory decoder (predicts all future timesteps at once)
        self.trajectory_decoder = nn.Linear(128, 60 * 3)  # 60 timesteps, each with x, y, heading
        
    def forward(self, image, motion_history):
        # Process image with ResNet backbone
        batch_size = image.size(0)
        x_visual = self.visual_backbone(image)
        x_visual = self.pool(x_visual)
        x_visual = x_visual.view(batch_size, -1)  # Flatten
        
        # Process motion history with GRU
        _, h_n = self.motion_gru(motion_history)
        # Take the last layer's hidden state
        x_motion = h_n[-1]
        
        # Concatenate visual and motion features
        combined = torch.cat([x_visual, x_motion], dim=1)
        
        # Fuse features
        fused = self.fusion(combined)
        
        # Decode trajectory
        trajectory = self.trajectory_decoder(fused)
        trajectory = trajectory.view(batch_size, 60, 3)  # Reshape to (batch_size, timesteps, features)
        
        return trajectory
    
# class SpatialAttention(nn.Module):
#     """
#     Spatial attention module that helps the model focus on relevant parts of the image
#     """
#     def __init__(self, in_channels):
#         super(SpatialAttention, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        
#     def forward(self, x):
#         # First convolution to reduce channels
#         attn = self.conv1(x)
#         attn = F.relu(attn)
        
#         # Second convolution to generate attention map
#         attn = self.conv2(attn)
        
#         # Apply sigmoid to get attention weights between 0 and 1
#         attn = torch.sigmoid(attn)
        
#         # Apply attention weights to input feature map
#         return x * attn.expand_as(x)

# class TemporalAttention(nn.Module):
#     """
#     Temporal attention module that helps the model focus on relevant timesteps
#     """
#     def __init__(self, hidden_size):
#         super(TemporalAttention, self).__init__()
#         self.attn = nn.Linear(hidden_size, 1)
        
#     def forward(self, gru_outputs):
#         # gru_outputs shape: [batch_size, seq_len, hidden_size]
        
#         # Calculate attention scores
#         attn_scores = self.attn(gru_outputs)  # [batch_size, seq_len, 1]
#         attn_scores = attn_scores.squeeze(-1)  # [batch_size, seq_len]
        
#         # Apply softmax to get attention weights
#         attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]
        
#         # Apply attention weights to get context vector
#         context = torch.bmm(attn_weights, gru_outputs)  # [batch_size, 1, hidden_size]
#         context = context.squeeze(1)  # [batch_size, hidden_size]
        
#         return context

# class CrossAttention(nn.Module):
#     """
#     Cross-attention module that helps relate visual features to motion features
#     """
#     def __init__(self, query_dim, key_dim, value_dim, attn_dim=64):
#         super(CrossAttention, self).__init__()
#         self.query_proj = nn.Linear(query_dim, attn_dim)
#         self.key_proj = nn.Linear(key_dim, attn_dim)
#         self.value_proj = nn.Linear(value_dim, value_dim)  # Usually we project values to same dim
#         self.scale = torch.sqrt(torch.tensor(attn_dim, dtype=torch.float32))
        
#     def forward(self, query, key, value):
#         # Project query, key, value
#         query_proj = self.query_proj(query)  # [batch_size, query_dim] -> [batch_size, attn_dim]
#         key_proj = self.key_proj(key)        # [batch_size, key_dim] -> [batch_size, attn_dim]
#         value_proj = self.value_proj(value)  # [batch_size, value_dim] -> [batch_size, value_dim]
        
#         # Calculate attention scores
#         attn_scores = torch.matmul(query_proj.unsqueeze(1), key_proj.unsqueeze(2)) / self.scale
#         attn_weights = F.softmax(attn_scores, dim=2)
        
#         # Apply attention weights to get context vector
#         context = attn_weights * value_proj.unsqueeze(1)
        
#         return context.squeeze(1)


# class ResNetFeatureExtractor(nn.Module):
#     """
#     ResNet-based feature extractor with partial fine-tuning
#     """
#     def __init__(self, output_dim=512, dropout_rate=0.3):
#         super(ResNetFeatureExtractor, self).__init__()
#         # Load pretrained ResNet18
#         resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
#         # Remove the final fully connected layer
#         self.features = nn.Sequential(*list(resnet.children())[:-2])
        
#         # Add a feature reduction layer to prevent overfitting
#         self.feature_reducer = nn.Sequential(
#             nn.Conv2d(512, output_dim, kernel_size=1),
#             nn.BatchNorm2d(output_dim),
#             nn.ReLU(),
#             nn.Dropout2d(dropout_rate)
#         )
        
#         # Spatial attention module to focus on relevant image regions
#         self.spatial_attention = SpatialAttention(output_dim)
        
#         # Adaptive pooling to get a fixed size output regardless of input dimensions
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # Freeze early layers (first 6 blocks)
#         self._freeze_early_layers()
        
#     def _freeze_early_layers(self):
#         # Freeze the first 6 blocks of ResNet (conv1, bn1, relu, maxpool, layer1, layer2)
#         for param in list(self.features.parameters())[:6]:
#             param.requires_grad = False
            
#     def forward(self, x):
#         x = self.features(x)
#         x = self.feature_reducer(x)
#         # Apply spatial attention - focusing on important regions of the feature map
#         x = self.spatial_attention(x)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)  # Flatten
#         return x


# class RoadMind(nn.Module):
#     def __init__(self, dropout_rate=0.3, resnet_output_dim=256):
#         super(RoadMind, self).__init__()
        
#         # ResNet feature extractor for visual processing
#         self.visual_extractor = ResNetFeatureExtractor(
#             output_dim=resnet_output_dim, 
#             dropout_rate=dropout_rate
#         )
        
#         # GRU for motion history processing
#         self.gru = nn.GRU(
#             input_size=3, 
#             hidden_size=64,
#             num_layers=2,
#             batch_first=True, 
#             dropout=dropout_rate if 2 > 1 else 0
#         )
        
#         # Temporal attention for GRU outputs
#         self.temporal_attention = TemporalAttention(64)
#         self.dropout_gru = nn.Dropout(dropout_rate)
        
#         # Cross-attention between visual and motion features
#         self.cross_attention = CrossAttention(
#             query_dim=64,                # GRU hidden size
#             key_dim=resnet_output_dim,   # ResNet output dimension
#             value_dim=resnet_output_dim  # ResNet output dimension
#         )
        
#         # Fusion and prediction with increased capacity
#         self.fusion = nn.Sequential(
#             nn.Linear(resnet_output_dim + 64, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
            
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate/2),
            
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate/4),
            
#             nn.Linear(128, 60 * 3)  # Predict 60 timesteps, each with x, y, heading
#         )
        
#         # Weight initialization for non-pretrained parts
#         self._initialize_weights()
           
#     def forward(self, image, motion_history):
#         # Process image with ResNet (batch_size, 3, 200, 300)
#         x_img = self.visual_extractor(image)
        
#         # Process motion history (batch_size, 21, 3) with GRU
#         gru_outputs, _ = self.gru(motion_history)
        
#         # Apply temporal attention to GRU outputs
#         x_mot = self.temporal_attention(gru_outputs)
#         x_mot = self.dropout_gru(x_mot)
        
#         # Apply cross-attention between motion and visual features
#         # Using motion as query and visual as key/value
#         attended_visual = self.cross_attention(x_mot, x_img, x_img)
        
#         # Concatenate features - using the attended visual features
#         combined = torch.cat([attended_visual, x_mot], dim=-1)
        
#         # Generate trajectory
#         trajectory = self.fusion(combined)
#         trajectory = trajectory.reshape(-1, 60, 3)  # Reshape to (batch_size, 60, 3)
        
#         return trajectory
    
#     def _initialize_weights(self):
#         """Initialize model weights for better training stability"""
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

### DO NOT MODIFY THIS MODEL ###
# class SpatialAttention(nn.Module):
#     """
#     Spatial attention module that helps the model focus on relevant parts of the image
#     """
#     def __init__(self, in_channels):
#         super(SpatialAttention, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        
#     def forward(self, x):
#         # First convolution to reduce channels
#         attn = self.conv1(x)
#         attn = F.relu(attn)
        
#         # Second convolution to generate attention map
#         attn = self.conv2(attn)
        
#         # Apply sigmoid to get attention weights between 0 and 1
#         attn = torch.sigmoid(attn)
        
#         # Apply attention weights to input feature map
#         return x * attn.expand_as(x)

# class TemporalAttention(nn.Module):
#     """
#     Temporal attention module that helps the model focus on relevant timesteps
#     """
#     def __init__(self, hidden_size):
#         super(TemporalAttention, self).__init__()
#         self.attn = nn.Linear(hidden_size, 1)
        
#     def forward(self, gru_outputs):
#         # gru_outputs shape: [batch_size, seq_len, hidden_size]
        
#         # Calculate attention scores
#         attn_scores = self.attn(gru_outputs)  # [batch_size, seq_len, 1]
#         attn_scores = attn_scores.squeeze(-1)  # [batch_size, seq_len]
        
#         # Apply softmax to get attention weights
#         attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]
        
#         # Apply attention weights to get context vector
#         context = torch.bmm(attn_weights, gru_outputs)  # [batch_size, 1, hidden_size]
#         context = context.squeeze(1)  # [batch_size, hidden_size]
        
#         return context

# class CrossAttention(nn.Module):
#     """
#     Cross-attention module that helps relate visual features to motion features
#     """
#     def __init__(self, query_dim, key_dim, value_dim, attn_dim=64):
#         super(CrossAttention, self).__init__()
#         self.query_proj = nn.Linear(query_dim, attn_dim)
#         self.key_proj = nn.Linear(key_dim, attn_dim)
#         self.value_proj = nn.Linear(value_dim, value_dim)  # Usually we project values to same dim
#         self.scale = torch.sqrt(torch.tensor(attn_dim, dtype=torch.float32))
        
#     def forward(self, query, key, value):
#         # Project query, key, value
#         query_proj = self.query_proj(query)  # [batch_size, query_dim] -> [batch_size, attn_dim]
#         key_proj = self.key_proj(key)        # [batch_size, key_dim] -> [batch_size, attn_dim]
#         value_proj = self.value_proj(value)  # [batch_size, value_dim] -> [batch_size, value_dim]
        
#         # Calculate attention scores
#         attn_scores = torch.matmul(query_proj.unsqueeze(1), key_proj.unsqueeze(2)) / self.scale
#         attn_weights = F.softmax(attn_scores, dim=2)
        
#         # Apply attention weights to get context vector
#         context = attn_weights * value_proj.unsqueeze(1)
        
#         return context.squeeze(1)

# class RoadMind(nn.Module):
#     def __init__(self, dropout_rate=0.3):
#         super(RoadMind, self).__init__()
        
#         # Increased channel capacity in convolutional layers
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.dropout1 = nn.Dropout2d(dropout_rate)
        
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.dropout2 = nn.Dropout2d(dropout_rate)
        
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.dropout3 = nn.Dropout2d(dropout_rate)
        
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.dropout4 = nn.Dropout2d(dropout_rate)
        
#         # Spatial attention after last conv layer
#         self.spatial_attention = SpatialAttention(128)
        
#         self.pool = nn.AdaptiveAvgPool2d((5, 5))
        
#         # Calculate flattened size
#         self.flatten_size = 128 * 5 * 5
        
#         # Using GRU instead of LSTM for more efficiency with smaller datasets
#         self.gru = nn.GRU(
#             input_size=3, 
#             hidden_size=64,
#             num_layers=2,
#             batch_first=True, 
#             dropout=dropout_rate if 2 > 1 else 0
#         )
        
#         # Temporal attention for GRU outputs
#         self.temporal_attention = TemporalAttention(64)
        
#         self.dropout_gru = nn.Dropout(dropout_rate)
        
#         # Cross-attention between visual and motion features
#         self.cross_attention = CrossAttention(
#             query_dim=64,    # GRU hidden size
#             key_dim=self.flatten_size,  # CNN feature size
#             value_dim=self.flatten_size  # CNN feature size
#         )
        
#         # Fusion and prediction with increased capacity
#         self.fusion = nn.Sequential(
#             nn.Linear(self.flatten_size + 64, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
            
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate/2),
            
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate/4),
            
#             nn.Linear(128, 60 * 3)  # Predict 60 timesteps, each with x, y, heading
#         )
        
#         # Weight initialization
#         self._initialize_weights()
           
#     def forward(self, image, motion_history):
#         # Process image (batch_size, 3, 200, 300)
#         x_img = F.relu(self.bn1(self.conv1(image)))
#         x_img = self.dropout1(x_img)
        
#         x_img = F.relu(self.bn2(self.conv2(x_img)))
#         x_img = self.dropout2(x_img)
        
#         x_img = F.relu(self.bn3(self.conv3(x_img)))
#         x_img = self.dropout3(x_img)
        
#         x_img = F.relu(self.bn4(self.conv4(x_img)))
#         x_img = self.dropout4(x_img)
        
#         # Apply spatial attention
#         x_img = self.spatial_attention(x_img)
        
#         x_img = self.pool(x_img)
#         x_img_flat = x_img.reshape(-1, self.flatten_size)
        
#         # Process motion history (batch_size, 21, 3) with GRU
#         gru_outputs, hidden = self.gru(motion_history)
        
#         # Apply temporal attention to GRU outputs
#         x_mot = self.temporal_attention(gru_outputs)
#         x_mot = self.dropout_gru(x_mot)
        
#         # Apply cross-attention between motion and visual features
#         # Using motion as query and visual as key/value
#         attended_visual = self.cross_attention(x_mot, x_img_flat, x_img_flat)
        
#         # Concatenate features - using the attended visual features
#         combined = torch.cat([attended_visual, x_mot], dim=-1)
        
#         # Generate trajectory
#         trajectory = self.fusion(combined)
#         trajectory = trajectory.reshape(-1, 60, 3)  # Reshape to (batch_size, 60, 3)
        
#         return trajectory
    
#     def _initialize_weights(self):
#         """Initialize model weights for better training stability"""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 nn.init.constant_(m.bias, 0)