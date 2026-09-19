import torch
import torch.nn as nn

class SensorCNN(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(SensorCNN,self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,out_channels=64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
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
        x = x.permute(0,2,1)
        features = self.conv_block(x)
        features = features.squeeze(-1)
        out = self.classifier(features)
        return out

class SensorLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(SensorLSTM,self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size,64),
            nn.ReLU(),
            nn.Linear(64,num_classes)
        )

    def forward(self,x):
        lstm_out , (h_n,c_n) = self.lstm(x)
        last_hidden_state = lstm_out[:,-1, :]
        out = self.classifier(last_hidden_state)
        return out