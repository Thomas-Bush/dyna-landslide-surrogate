import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Define the encoder path with the corresponding channels
        self.enc1 = self.contracting_block(in_channels, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.enc3 = self.contracting_block(128, 256)
        self.enc4 = self.contracting_block(256, 512)
        
        # Define the bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Define the decoder path with the corresponding channels
        self.de3 = self.expansive_block(1024, 512, 256)
        self.up2 = self.expansive_block(512, 256, 128)
        self.up1 = self.expansive_block(256, 128, 64)
        
        # Final output layer
        self.final_out = nn.Conv2d(128, out_channels, kernel_size=1)

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channel, out_channels, kernel_size=3, stride=2, 
                               padding=1, output_padding=1)
        )
        return block

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        # Bottleneck
        x5 = self.bottleneck(x4)
        
        # Decoder with skip connections
        x = self.up3(torch.cat([x5, x4], 1))
        x = self.up2(torch.cat([x, x3], 1))
        x = self.up1(torch.cat([x, x2], 1))
        
        # Final output layer
        out = self.final_out(torch.cat([x, x1], 1))
        return out


class CNNLSTMModel(nn.Module):
    """CNN-LSTM model for learning the sequence of landslide debris travel."""
    def __init__(self, in_channels, lstm_hidden_size, lstm_layers=1, n_classes=1):
        super(CNNLSTMModel, self).__init__()
        self.unet =UNet(n_channels=in_channels, n_classes=n_classes)
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)
        self.final_fc = nn.Linear(lstm_hidden_size, n_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        # Flatten the timesteps into the batch dimension
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.unet(c_in)
        # Reconstruct the temporal dimension
        c_out = c_out.view(batch_size, timesteps, -1)

        # Pass features through LSTM
        lstm_out, (h_n, c_n) = self.lstm(c_out)
        # Take only the output of the last LSTM cell
        final_output = self.final_fc(lstm_out[:, -1])
        return final_output