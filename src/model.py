import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# class TemporalAttention(nn.Module):
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

# class DrivingSceneEncoder(nn.Module):
#     def __init__(self, output_dim=128, dropout_rate=0.2):
#         super(DrivingSceneEncoder, self).__init__() 
        
#         self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
#         # Remove the classifier and pooling layers
#         self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        
#         self.fc = nn.Sequential(
#             nn.Linear(1280, output_dim),  # EfficientNet-B0 outputs 1280 features
#             nn.ReLU(),
#             nn.Dropout(dropout_rate)
#         )
        
#         # for name, param in list(self.backbone.features.named_parameters())[:4]:
#         #     param.requires_grad = False
        
#     def forward(self, x, return_features=False):
#         # Ensure input has the right dimensions
#         if x.dim() == 3:
#             x = x.unsqueeze(0)
        
#         # Pass through EfficientNet backbone
#         features = self.backbone(x)
        
#         # Global average pooling
#         x = self.adaptive_pool(features) # [batch_size, 1280, 1, 1] 
#         x = x.view(x.size(0), -1)  # [batch_size, 1280]
#         x = self.fc(x)
#         if return_features:
#             return x, features
#         return x

# class RoadMind(nn.Module):
#     def __init__(self, input_dim=3, hidden_dim=128, image_embed_dim=128, output_seq_len=60, 
#                  num_layers=2, dropout_rate=0.3, bidirectional=True,
#                  use_depth_aux=False, use_semantic_aux=False, num_semantic_classes=15):
#         super(RoadMind, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_seq_len = output_seq_len
#         self.bidirectional = bidirectional
#         self.num_directions = 2 if bidirectional else 1
        
#         # Auxiliary task flags
#         self.use_depth_aux = use_depth_aux
#         self.use_semantic_aux = use_semantic_aux
        
        
#         # EfficientNet-based encoder for processing camera data
#         image_embed_dim = image_embed_dim * self.num_directions
#         self.image_encoder = DrivingSceneEncoder(output_dim=image_embed_dim, dropout_rate=dropout_rate)
        
#         # GRU layer for processing history features
#         self.gru = nn.GRU(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout_rate if num_layers > 1 else 0,
#             bidirectional=bidirectional
#         )
        
#         # Hidden state dimension (accounting for bidirectionality and num_layers)
#         hidden_state_dim = hidden_dim * self.num_directions
        
#         # Temporal attention mechanism
#         self.attention = TemporalAttention(hidden_state_dim)
        
#         # Fusion layer to combine image features, attended trajectory features, and hidden state
#         self.fusion = nn.Sequential(
#             nn.Linear(hidden_state_dim * 2 + image_embed_dim, hidden_state_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate)
#         )
        
#         # Highway connection for better gradient flow
#         self.highway_gate = nn.Sequential(
#             nn.Linear(hidden_state_dim * 2 + image_embed_dim, hidden_state_dim),
#             nn.Sigmoid()
#         )
        
#         # MLP for decoding
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_state_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim, output_seq_len * input_dim)
#         )
        
#         # Depth estimation decoder (if enabled)
#         if self.use_depth_aux:
#             self.depth_decoder = nn.ModuleList([
#                 nn.ConvTranspose2d(1280, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 nn.ReLU(),
#                 nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 nn.ReLU(),
#                 nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 nn.ReLU(),
#                 nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 nn.ReLU(),
#                 nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(32, 1, kernel_size=3, padding=1),
#                 nn.ReLU(),
#                 # Add a final upsample to ensure correct dimensions
#                 nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
#             ])

#         if self.use_semantic_aux:
#             self.semantic_decoder = nn.ModuleList([
#                 nn.ConvTranspose2d(1280, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 nn.ReLU(),
#                 nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 nn.ReLU(),
#                 nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 nn.ReLU(),
#                 nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 nn.ReLU(),
#                 nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(32, num_semantic_classes, kernel_size=3, padding=1),
#                 # Add a final upsample to ensure correct dimensions
#                 nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
#             ])
#         self._initialize_weights()
        
#     def forward(self, camera, history_features):
        
#         # Process camera through custom image encoder
#         image_features, self.cnn_features = self.image_encoder(camera, return_features=True)
        
#         # Process history features through GRU
#         gru_out, hidden = self.gru(history_features)
        
#         # Apply temporal attention to GRU outputs
#         attended_features = self.attention(gru_out)
        
#         # Get the final hidden state
#         if self.bidirectional:
#             # For bidirectional, combine the last hidden states from both directions
#             last_layer_hidden = hidden[-2:].transpose(0, 1).contiguous()
#             final_hidden = last_layer_hidden.view(last_layer_hidden.size(0), -1)
#         else:
#             # For unidirectional, just take the last hidden state
#             final_hidden = hidden[-1]
            
#         # Combine the image features, attended features, and final hidden state
#         combined = torch.cat([image_features, attended_features, final_hidden], dim=1)
        
#         # Highway connection
#         gate = self.highway_gate(combined)
#         transformed = self.fusion(combined)
#         fused = gate * transformed + (1 - gate) * final_hidden
        
#         # Decode to get predicted trajectory
#         output = self.decoder(fused)
        
#         # Reshape to match the expected output format
#         output = output.view(-1, self.output_seq_len, self.input_dim)
        
#         # Initialize auxiliary outputs as None
#         depth_output = None
#         semantic_output = None
#         print(self.cnn_features.shape)
#         # Generate depth prediction if enabled
#         if self.use_depth_aux:
#             depth_features = self.cnn_features  # Use the saved CNN features
#             for layer in self.depth_decoder:
#                 depth_features = layer(depth_features)
#             # Permute from [B, C, H, W] to [B, H, W, C] to match ground truth format
#             depth_output = depth_features.permute(0, 2, 3, 1)
        
#         # Generate semantic segmentation if enabled
#         if self.use_semantic_aux:
#             semantic_features = self.cnn_features  # Use the saved CNN features
#             for layer in self.semantic_decoder:
#                 semantic_features = layer(semantic_features)
#             semantic_output = semantic_features
        
#         return output, depth_output, semantic_output
        

#     def _initialize_weights(self):
#         # Initialize weights for non-pretrained parts of the model
#         for m in [self.fusion, self.highway_gate, self.decoder, self.attention, self.depth_decoder, self.semantic_decoder]:
#             for layer in m.modules():
#                 if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                     # Kaiming/He initialization works well with ReLU
#                     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                     if m.bias is not None:
#                         nn.init.constant_(m.bias, 0)
                        
#                 elif isinstance(m, nn.BatchNorm2d):
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)
                        
#                 elif isinstance(layer, nn.Linear):
#                     # Use kaiming for hidden layers with ReLU
#                     if layer != self.decoder[-1]:  # Not the final output layer
#                         nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
#                         if layer.bias is not None:
#                             nn.init.constant_(layer.bias, 0)
#                     else:  # Output layer gets Xavier/Glorot for better distribution
#                         nn.init.xavier_normal_(layer.weight)
#                         if layer.bias is not None:
#                             nn.init.constant_(layer.bias, 0)
                    
#         # Special initialization for GRU
#         for name, param in self.gru.named_parameters():
#             if 'weight_ih' in name:  # Input-to-hidden weights
#                 nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
#             elif 'weight_hh' in name:  # Hidden-to-hidden (recurrent) weights
#                 nn.init.orthogonal_(param)  # Orthogonal initialization for recurrent connections
#             elif 'bias' in name:  # Biases
#                 nn.init.constant_(param, 0)

class TemporalAttention(nn.Module):
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

class DrivingSceneEncoder(nn.Module):
    def __init__(self, input_channels=3, output_dim=128, dropout_rate=0.2):
        super(DrivingSceneEncoder, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Third convolutional block 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # Fourth convolutional block 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        # Global average pooling to reduce spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        
        # Final FC layers
        self.fc1 = nn.Linear(256, output_dim)
        self.dropout_fc = nn.Dropout(dropout_rate)
        
    def forward(self, x, return_features=False):
        # Ensure input has the right dimensions
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        
        # Third block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        
        # Fourth block
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout4(x)
        
        # Save features for auxiliary tasks before pooling
        cnn_features = x  # Shape: [B, 256, H/16, W/16]

        # Use adaptive pooling to get a fixed size output
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 256)
        
        # Final fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        
        if return_features:
            return x, cnn_features
        else:
            return x

class RoadMind(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, image_embed_dim=128, output_seq_len=60, 
                 num_layers=2, dropout_rate=0.3, bidirectional=True, 
                 use_depth_aux=False, use_semantic_aux=False, num_semantic_classes=15):
        super(RoadMind, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_seq_len = output_seq_len
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Auxiliary task flags
        self.use_depth_aux = use_depth_aux
        self.use_semantic_aux = use_semantic_aux
        
        # Custom CNN encoder for processing camera data
        self.image_encoder = DrivingSceneEncoder(input_channels=3, output_dim=image_embed_dim, dropout_rate=dropout_rate)
        
        # Save CNN features for auxiliary tasks
        self.cnn_features = None
        
        # GRU layer for processing history features
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Hidden state dimension (accounting for bidirectionality and num_layers)
        hidden_state_dim = hidden_dim * self.num_directions
        
        # Temporal attention mechanism
        self.attention = TemporalAttention(hidden_state_dim)
        
        # Fusion layer to combine image features, attended trajectory features, and hidden state
        self.fusion = nn.Sequential(
            nn.Linear(hidden_state_dim * 2 + image_embed_dim, hidden_state_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Highway connection for better gradient flow
        self.highway_gate = nn.Sequential(
            nn.Linear(hidden_state_dim * 2 + image_embed_dim, hidden_state_dim),
            nn.Sigmoid()
        )
        
        # MLP for decoding
        self.decoder = nn.Sequential(
            nn.Linear(hidden_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_seq_len * input_dim)
        )
        
        # Depth estimation decoder (if enabled)
        if self.use_depth_aux:
            self.depth_decoder = nn.ModuleList([
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=3, padding=1),
                nn.ReLU(),
                # Add a final upsample to ensure correct dimensions
                nn.Upsample(size=(200, 300), mode='bilinear', align_corners=False)
            ])
        
        # Semantic segmentation decoder (if enabled)
        if self.use_semantic_aux:
            self.semantic_decoder = nn.ModuleList([
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.Conv2d(16, num_semantic_classes, kernel_size=3, padding=1),
                # Add a final upsample to ensure correct dimensions
                nn.Upsample(size=(200, 300), mode='bilinear', align_corners=False)
            ])
        
        self._initialize_weights()
        
    def forward(self, camera, history_features):
        B = camera.size(0)
        
        # Process camera through image encoder - need to modify DrivingSceneEncoder to return intermediate features
        image_features, self.cnn_features = self.image_encoder(camera, return_features=True)
        
        # Process history features through GRU
        gru_out, hidden = self.gru(history_features)
        
        # Apply temporal attention to GRU outputs
        attended_features = self.attention(gru_out)
        
        # Get the final hidden state
        if self.bidirectional:
            # For bidirectional, combine the last hidden states from both directions
            last_layer_hidden = hidden[-2:].transpose(0, 1).contiguous()
            final_hidden = last_layer_hidden.view(last_layer_hidden.size(0), -1)
        else:
            # For unidirectional, just take the last hidden state
            final_hidden = hidden[-1]
        
        # Combine the image features, attended features, and final hidden state
        combined = torch.cat([image_features, attended_features, final_hidden], dim=1)
        
        # Highway connection
        gate = self.highway_gate(combined)
        transformed = self.fusion(combined)
        fused = gate * transformed + (1 - gate) * final_hidden
        
        # Decode to get predicted trajectory
        output = self.decoder(fused)
        
        # Reshape to match the expected output format
        output = output.view(-1, self.output_seq_len, self.input_dim)
        
        # Initialize auxiliary outputs as None
        depth_output = None
        semantic_output = None
        
        # Generate depth prediction if enabled
        if self.use_depth_aux:
            depth_features = self.cnn_features  # Use the saved CNN features
            for layer in self.depth_decoder:
                depth_features = layer(depth_features)
            depth_output = depth_features
        
        # Generate semantic segmentation if enabled
        if self.use_semantic_aux:
            semantic_features = self.cnn_features  # Use the saved CNN features
            for layer in self.semantic_decoder:
                semantic_features = layer(semantic_features)
            semantic_output = semantic_features
        
        return output, depth_output, semantic_output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # Kaiming/He initialization works well with ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Linear):
                # Use kaiming for hidden layers with ReLU
                if m != self.decoder[-1]:  # Not the final output layer
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)
                else:  # Output layer gets Xavier/Glorot for better distribution
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                    
        # Special initialization for GRU
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:  # Input-to-hidden weights
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'weight_hh' in name:  # Hidden-to-hidden (recurrent) weights
                nn.init.orthogonal_(param)  # Orthogonal initialization for recurrent connections
            elif 'bias' in name:  # Biases
                nn.init.constant_(param, 0)
