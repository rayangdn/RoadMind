import torch
import torch.nn as nn
import torch.nn.functional as F
import math as math

### BASELINE MODEL DO NOT MODIFY
# class RoadMind(nn.Module):
#     def __init__(self):
#         super(RoadMind, self).__init__()
        
#         # Image processing
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
#         self.pool = nn.AdaptiveAvgPool2d((5, 5))
        
#         # Calculate flattened size
#         self.flatten_size = 64 * 5 * 5
        
#         # Command embedding
#         self.command_embedding = nn.Embedding(3, 8)
        
#         # Motion history processing
#         self.lstm = nn.LSTM(input_size=3, hidden_size=32, num_layers=1, batch_first=True)
        
#         # Fusion and prediction
#         self.fusion = nn.Sequential(
#             nn.Linear(self.flatten_size + 8 + 32, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 60 * 3)  # Predict 60 timesteps, each with x, y, heading
#         )
        
#         self.init_weights()
        
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
                    
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
#                 if m.bias is not None:
#                     # Calculate proper gain for bias initialization
#                     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
#                     bound = 1 / math.sqrt(fan_in)
#                     nn.init.uniform_(m.bias, -bound, bound)
                    
#             elif isinstance(m, nn.LSTM):
#                 # Special initialization for LSTM parameters
#                 for name, param in m.named_parameters():
#                     if 'weight_ih' in name:
#                         # Input-to-hidden weights
#                         nn.init.xavier_uniform_(param.data)
#                     elif 'weight_hh' in name:
#                         # Hidden-to-hidden (recurrent) weights - orthogonal initialization helps with vanishing gradients
#                         nn.init.orthogonal_(param.data)
#                     elif 'bias' in name:
#                         # Bias parameters
#                         nn.init.constant_(param.data, 0)
#                         # Forget gate bias initialization to 1 (helps remember longer-term dependencies)
#                         param.data[m.hidden_size:2*m.hidden_size].fill_(1)
                        
#             elif isinstance(m, nn.BatchNorm2d):
#                 # Default initialization for batch normalization
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
        
#     def forward(self, image, command, motion_history):
#         # Process image (batch_size, 3, 200, 300)
#         x_img = F.relu(self.conv1(image))
#         x_img = F.relu(self.conv2(x_img))
#         x_img = F.relu(self.conv3(x_img))
#         x_img = self.pool(x_img)
#         x_img = x_img.reshape(-1, self.flatten_size)
        
#         # Process command (batch_size,)
#         # Command should be 0, 1, or 2 (forward, left, right)
#         x_cmd = self.command_embedding(command)
#         x_cmd = x_cmd.squeeze(1)
        
#         # Process motion history (batch_size, 21, 3)
#         _, (x_mot, _) = self.lstm(motion_history)
#         x_mot = x_mot.squeeze(0)
        
#         # Concatenate features
#         combined = torch.cat([x_img, x_cmd, x_mot], dim=-1)
        
#         # Generate trajectory
#         trajectory = self.fusion(combined)
#         trajectory = trajectory.reshape(-1, 60, 3)  # Reshape to (batch_size, 60, 3)
        
#         return trajectory
###########################################################

