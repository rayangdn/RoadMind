import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# class TemporalAttention(nn.Module):
#     def __init__(self, hidden_dim):
#         super(TemporalAttention, self).__init__()
#         self.query_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.key_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.value_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
#     def forward(self, query, keys):
#         # Move scale tensor to the same device as input
#         self.scale = self.scale.to(query.device)
        
#         # Project query, keys and values
#         batch_size = query.shape[0]
#         seq_len = keys.shape[1]
        
#         # Reshape query for broadcasting: [batch_size, 1, hidden_dim]
#         query = self.query_proj(query).unsqueeze(1)
        
#         # Project keys: [batch_size, seq_len, hidden_dim]
#         keys = self.key_proj(keys)
        
#         # Project values: [batch_size, seq_len, hidden_dim]
#         values = self.value_proj(keys)
        
#         # Calculate attention scores
#         # [batch_size, 1, hidden_dim] x [batch_size, hidden_dim, seq_len] = [batch_size, 1, seq_len]
#         attention = torch.bmm(query, keys.transpose(1, 2)) / self.scale
        
#         # Apply softmax to get attention weights
#         attention_weights = F.softmax(attention, dim=2)
        
#         # Apply attention weights to values
#         # [batch_size, 1, seq_len] x [batch_size, seq_len, hidden_dim] = [batch_size, 1, hidden_dim]
#         context = torch.bmm(attention_weights, values).squeeze(1)
        
#         return context, attention_weights.squeeze(1)


# class CrossModalAttention(nn.Module):
#     def __init__(self, traj_dim, visual_dim):
#         super(CrossModalAttention, self).__init__()
#         self.traj_proj = nn.Linear(traj_dim, 256)
#         self.visual_proj = nn.Linear(visual_dim, 256)
#         self.attention = nn.Linear(256, 1)
        
#     def forward(self, traj_features, visual_features):
#         # Project features to common space
#         print(traj_features.shape, visual_features.shape)
#         traj_proj = self.traj_proj(traj_features)
#         visual_proj = self.visual_proj(visual_features)
        
#         # Calculate attention weights
#         traj_score = self.attention(traj_proj)
#         visual_score = self.attention(visual_proj)
        
#         # Apply softmax to get attention weights
#         weights = F.softmax(torch.cat([traj_score, visual_score], dim=1), dim=1)
        
#         # Weighted sum of features
#         fused_features = weights[:, 0:1] * traj_features + weights[:, 1:2] * visual_features
        
#         return fused_features


# class RoadMind(nn.Module):
#     def __init__(self, history_dim=3, hidden_dim=128, future_dim=3, future_steps=60, 
#                  num_gru_layers=2, dropout_rate=0.3, bidirectional=True):
#         super(RoadMind, self).__init__()
        
#         self.hidden_dim = hidden_dim
#         self.future_dim = future_dim
#         self.future_steps = future_steps
#         self.bidirectional = bidirectional
        
#         # Visual feature extraction - ResNet18 with frozen early layers
#         self.visual_encoder = self._get_visual_encoder()
#         visual_feat_dim = 512  # Output dimension of ResNet18 features
        
#         # Global average pooling for visual features
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
        
#         # GRU for processing motion history
#         self.gru = nn.GRU(
#             input_size=history_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_gru_layers,
#             batch_first=True,
#             dropout=dropout_rate if num_gru_layers > 1 else 0,
#             bidirectional=bidirectional
#         )
        
#         # Calculate GRU output dimension (doubled if bidirectional)
#         gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
#         # Temporal attention mechanism for trajectory
#         self.temporal_attention = TemporalAttention(gru_output_dim)
        
#         # Cross-modal attention between trajectory and visual features
#         #self.cross_modal_attention = CrossModalAttention(gru_output_dim, visual_feat_dim)
        
#         # MLP decoder for predicting future trajectory
#         self.decoder = nn.Sequential(
#             nn.Linear(gru_output_dim + visual_feat_dim, 256),  # Combined features
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(128, future_steps * future_dim)
#         )
        
#     def _get_visual_encoder(self):
#         resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
#         # Remove the final fully connected layer
#         modules = list(resnet.children())[:-2]  # Remove avg pool and fc
#         encoder = nn.Sequential(*modules)
        
#         # Freeze early layers (first 6 blocks)
#         for i, child in enumerate(encoder.children()):
#             if i < 6: 
#                 for param in child.parameters():
#                     param.requires_grad = False
                    
#         return encoder
        
#     def forward(self, camera, sdc_history_feature, return_attention=False):
#         batch_size = sdc_history_feature.size(0)
        
#         # Process camera images - output shape: [batch_size, 512, H, W]
#         visual_features = self.visual_encoder(camera)
        
#         # Global average pooling to get a vector - output shape: [batch_size, 512, 1, 1]
#         visual_features = self.global_pool(visual_features)
        
#         # Flatten to [batch_size, 512]
#         visual_features = visual_features.view(batch_size, -1)
        
#         # Process motion history through GRU
#         gru_outputs, h_n = self.gru(sdc_history_feature)
        
#         # Extract final hidden state from all layers/directions
#         if self.bidirectional:
#             # Concatenate forward and backward final hidden states from the last layer
#             h_n_final = torch.cat([h_n[-2], h_n[-1]], dim=1)
#         else:
#             # Take the final hidden state from the last layer
#             h_n_final = h_n[-1] 
        
#         # Apply temporal attention over the GRU outputs using the final hidden state as query
#         context, temporal_attention_weights = self.temporal_attention(h_n_final, gru_outputs)
        
#         # Apply cross-modal attention between trajectory and visual features
#         #fused_features = self.cross_modal_attention(context, visual_features)
        
#         # Concatenate GRU output and visual features
#         fused_features = torch.cat([context, visual_features], dim=1)
        
#         # Decode the combined representation to predict future trajectory
#         future_trajectory = self.decoder(fused_features)
        
#         # Reshape to (batch_size, future_steps, future_dim)
#         future_trajectory = future_trajectory.view(batch_size, self.future_steps, self.future_dim)
        
#         if return_attention:
#             return future_trajectory, temporal_attention_weights
#         else:
#             return future_trajectory
        
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models


# class TemporalAttention(nn.Module):
#     def __init__(self, hidden_dim):
#         super(TemporalAttention, self).__init__()
#         self.query_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.key_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.value_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
#     def forward(self, query, keys):
#         # Move scale tensor to the same device as input
#         self.scale = self.scale.to(query.device)
        
#         # Project query, keys and values
#         batch_size = query.shape[0]
#         seq_len = keys.shape[1]
        
#         # Reshape query for broadcasting: [batch_size, 1, hidden_dim]
#         query = self.query_proj(query).unsqueeze(1)
        
#         # Project keys: [batch_size, seq_len, hidden_dim]
#         keys = self.key_proj(keys)
        
#         # Project values: [batch_size, seq_len, hidden_dim]
#         values = self.value_proj(keys)
        
#         # Calculate attention scores
#         # [batch_size, 1, hidden_dim] x [batch_size, hidden_dim, seq_len] = [batch_size, 1, seq_len]
#         attention = torch.bmm(query, keys.transpose(1, 2)) / self.scale
        
#         # Apply softmax to get attention weights
#         attention_weights = F.softmax(attention, dim=2)
        
#         # Apply attention weights to values
#         # [batch_size, 1, seq_len] x [batch_size, seq_len, hidden_dim] = [batch_size, 1, hidden_dim]
#         context = torch.bmm(attention_weights, values).squeeze(1)
        
#         return context, attention_weights.squeeze(1)


# class CrossModalAttention(nn.Module):
#     def __init__(self, traj_dim, visual_dim):
#         super(CrossModalAttention, self).__init__()
#         # Project both inputs to the same embedding dimension
#         self.embedding_dim = 256
#         self.traj_proj = nn.Linear(traj_dim, self.embedding_dim)
#         self.visual_proj = nn.Linear(visual_dim, self.embedding_dim)
        
#         # Attention scoring layer
#         self.attention = nn.Linear(self.embedding_dim, 1)
        
#         # Output projection to handle different input dimensions
#         # We'll project to the larger of the two dimensions
#         self.output_dim = max(traj_dim, visual_dim)
#         self.output_proj = nn.Sequential(
#             nn.Linear(self.embedding_dim, self.output_dim),
#             nn.ReLU()
#         )
        
#     def forward(self, traj_features, visual_features):
#         # Project features to common embedding space
#         traj_proj = self.traj_proj(traj_features)
#         visual_proj = self.visual_proj(visual_features)
        
#         # Calculate attention weights
#         traj_score = self.attention(traj_proj)
#         visual_score = self.attention(visual_proj)
        
#         # Apply softmax to get attention weights
#         weights = F.softmax(torch.cat([traj_score, visual_score], dim=1), dim=1)
        
#         # Weighted sum in embedding space
#         combined_embedding = weights[:, 0:1] * traj_proj + weights[:, 1:2] * visual_proj
        
#         # Project back to output dimension
#         fused_features = self.output_proj(combined_embedding)
        
#         return fused_features


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
    def forward(self, query, keys):
        # Move scale tensor to the same device as input
        self.scale = self.scale.to(query.device)
        
        # Project query, keys and values
        batch_size = query.shape[0]
        seq_len = keys.shape[1]
        
        # Reshape query for broadcasting: [batch_size, 1, hidden_dim]
        query = self.query_proj(query).unsqueeze(1)
        
        # Project keys: [batch_size, seq_len, hidden_dim]
        keys = self.key_proj(keys)
        
        # Project values: [batch_size, seq_len, hidden_dim]
        values = self.value_proj(keys)
        
        # Calculate attention scores
        # [batch_size, 1, hidden_dim] x [batch_size, hidden_dim, seq_len] = [batch_size, 1, seq_len]
        attention = torch.bmm(query, keys.transpose(1, 2)) / self.scale
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention, dim=2)
        
        # Apply attention weights to values
        # [batch_size, 1, seq_len] x [batch_size, seq_len, hidden_dim] = [batch_size, 1, hidden_dim]
        context = torch.bmm(attention_weights, values).squeeze(1)
        
        return context, attention_weights.squeeze(1)


class CrossModalAttention(nn.Module):
    def __init__(self, traj_dim, visual_dim):
        super(CrossModalAttention, self).__init__()
        # Project both inputs to the same embedding dimension
        self.embedding_dim = 256
        self.traj_proj = nn.Linear(traj_dim, self.embedding_dim)
        self.visual_proj = nn.Linear(visual_dim, self.embedding_dim)
        
        # Attention scoring layer
        self.attention = nn.Linear(self.embedding_dim, 1)
        
        # Output projection to handle different input dimensions
        # We'll project to the larger of the two dimensions
        self.output_dim = max(traj_dim, visual_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.output_dim),
            nn.ReLU()
        )
        
    def forward(self, traj_features, visual_features):
        # Project features to common embedding space
        traj_proj = self.traj_proj(traj_features)
        visual_proj = self.visual_proj(visual_features)
        
        # Calculate attention weights
        traj_score = self.attention(traj_proj)
        visual_score = self.attention(visual_proj)
        
        # Apply softmax to get attention weights
        weights = F.softmax(torch.cat([traj_score, visual_score], dim=1), dim=1)
        
        # Weighted sum in embedding space
        combined_embedding = weights[:, 0:1] * traj_proj + weights[:, 1:2] * visual_proj
        
        # Project back to output dimension
        fused_features = self.output_proj(combined_embedding)
        
        return fused_features


class RoadMind(nn.Module):
    def __init__(self, history_dim=3, hidden_dim=128, future_dim=3, future_steps=60, 
                 num_gru_layers=2, dropout_rate=0.3, bidirectional=True):
        super(RoadMind, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.future_dim = future_dim
        self.future_steps = future_steps
        self.bidirectional = bidirectional
        
        # Visual feature extraction - EfficientNet-Lite0 (most lightweight)
        self.visual_encoder = self._get_visual_encoder()
        visual_feat_dim = 1280  # Output dimension of EfficientNet features
        
        # Global average pooling for visual features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # GRU for processing motion history
        self.gru = nn.GRU(
            input_size=history_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout_rate if num_gru_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate GRU output dimension (doubled if bidirectional)
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Temporal attention mechanism for trajectory
        self.temporal_attention = TemporalAttention(gru_output_dim)
        
        # Cross-modal attention between trajectory and visual features
        self.cross_modal_attention = CrossModalAttention(gru_output_dim, visual_feat_dim)
        
        # Output dimension from cross-modal attention
        cross_modal_output_dim = max(gru_output_dim, visual_feat_dim)
        
        # MLP decoder for predicting future trajectory
        self.decoder = nn.Sequential(
            nn.Linear(cross_modal_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, future_steps * future_dim)
        )
        
    def _get_visual_encoder(self):
        # Use EfficientNet-Lite0 (the most lightweight version for mobile)
        # Note: In torchvision, this is accessed through efficientnet_b0 with a different weight set
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Create a smaller, more efficient feature extractor
        # 1. Keep the first part of the network (features)
        # 2. Remove the classifier (final layer)
        encoder = nn.Sequential(*list(efficientnet.children())[:-1])
        
        # Add a 1x1 convolution to reduce the channel dimensions for better efficiency
        # This reduces the feature dimension from 1280 to 640
        feature_reducer = nn.Conv2d(1280, 640, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Combine the encoder and feature reducer
        efficient_encoder = nn.Sequential(
            encoder,
            feature_reducer
        )
        
        # Freeze early features to prevent overfitting and reduce training time
        # Freeze first 3 blocks
        features_to_freeze = list(encoder[0].children())[:3]
        for block in features_to_freeze:
            for param in block.parameters():
                param.requires_grad = False
                    
        return efficient_encoder
        
    def forward(self, camera, sdc_history_feature, return_attention=False):
        batch_size = sdc_history_feature.size(0)
        
        # Process camera images - output shape will be [batch_size, 640, H, W]
        visual_features = self.visual_encoder(camera)
        
        # Global average pooling to get a vector - output shape: [batch_size, 640, 1, 1]
        visual_features = self.global_pool(visual_features)
        
        # Flatten to [batch_size, 640]
        visual_features = visual_features.view(batch_size, -1)
        
        # Process motion history through GRU
        gru_outputs, h_n = self.gru(sdc_history_feature)
        
        # Extract final hidden state from all layers/directions
        if self.bidirectional:
            # Concatenate forward and backward final hidden states from the last layer
            h_n_final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            # Take the final hidden state from the last layer
            h_n_final = h_n[-1]
        
        # Apply temporal attention over the GRU outputs using the final hidden state as query
        context, temporal_attention_weights = self.temporal_attention(h_n_final, gru_outputs)
        
        # Apply cross-modal attention between trajectory and visual features
        fused_features = self.cross_modal_attention(context, visual_features)
        
        # Decode the combined representation to predict future trajectory
        future_trajectory = self.decoder(fused_features)
        
        # Reshape to (batch_size, future_steps, future_dim)
        future_trajectory = future_trajectory.view(batch_size, self.future_steps, self.future_dim)
        
        if return_attention:
            return future_trajectory, temporal_attention_weights
        else:
            return future_trajectory
        
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
        
#         #combined = self.transformer_encoder(combined.unsqueeze(1)).squeeze(1)
        
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