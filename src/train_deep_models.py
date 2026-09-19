import torch
import torch.nn as nn

class SensorCNN(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(SensorCNN,self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,out_channels=64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64,out_channel=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self,x):
        features = self.conv_block(x)
        features = features.squeeze(-1)
        out = self.classifier(features)
        return out