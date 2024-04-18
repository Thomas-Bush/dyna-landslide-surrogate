import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


### STANDALONE CNNs ###

class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Input: 3 channels, Output: 32 channels
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 2, kernel_size=3, padding=1)  # Output: 2 channels (velocity and thickness)

    def forward(self, x):
        # Encoder path
        x1 = F.relu(self.bn1(self.conv1(x)))     # 64x64x32
        x2 = self.pool(x1)                       # 32x32x32
        x3 = F.relu(self.bn2(self.conv2(x2)))    # 32x32x64
        x4 = self.pool(x3)                       # 16x16x64
        x5 = F.relu(self.bn3(self.conv3(x4)))    # 16x16x128
        x6 = self.pool(x5)                       # 8x8x128
        x7 = F.relu(self.bn4(self.conv4(x6)))    # 8x8x256

        # Decoder path
        x8 = self.up(x7)                         # 16x16x256
        x9 = F.relu(self.bn5(self.conv5(x8)))    # 16x16x128
        x10 = self.up(x9)                        # 32x32x128
        x11 = F.relu(self.bn6(self.conv6(x10)))  # 32x32x64
        x12 = self.up(x11)                       # 64x64x64

        # Output layer
        x13 = self.conv7(x12)                    # 64x64x2
        return x13
    
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()

        # Define the U-Net architecture
        # Contracting Path (Encoder)
        self.enc_conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)

        # Expanding Path (Decoder)
        self.up_conv0 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.dec_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)  # Output: 1 channel (velocity)

    def forward(self, x):
        # Encoder
        enc0 = F.relu(self.enc_conv0(x))
        enc1 = F.relu(self.enc_conv1(self.pool(enc0)))
        enc2 = F.relu(self.enc_conv2(self.pool(enc1)))
        enc3 = F.relu(self.enc_conv3(self.pool(enc2)))

        # Decoder
        dec2 = F.relu(self.dec_conv1(torch.cat((self.up_conv0(enc3), enc2), dim=1)))
        dec1 = F.relu(self.dec_conv2(torch.cat((self.up_conv1(dec2), enc1), dim=1)))
        dec0 = F.relu(self.dec_conv3(torch.cat((self.up_conv2(dec1), enc0), dim=1)))

        # Final convolution
        return self.final_conv(dec0)
    
class LargeUNet(nn.Module):
    def __init__(self):
        super(LargeUNet, self).__init__()

        # Contracting Path (Encoder)
        self.enc_conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(64)
        self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.enc_conv4 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(1024)

        self.pool = nn.MaxPool2d(2, 2)

        # Expanding Path (Decoder)
        self.up_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv0 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.dec_conv4 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.dec_bn4 = nn.BatchNorm2d(512)
        self.dec_conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(256)
        self.dec_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(128)
        self.dec_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(64)

        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc0 = F.relu(self.bn0(self.enc_conv0(x)))
        enc1 = F.relu(self.bn1(self.enc_conv1(self.pool(enc0))))
        enc2 = F.relu(self.bn2(self.enc_conv2(self.pool(enc1))))
        enc3 = F.relu(self.bn3(self.enc_conv3(self.pool(enc2))))
        enc4 = F.relu(self.bn4(self.enc_conv4(self.pool(enc3))))

        # Decoder
        dec3 = F.relu(self.dec_bn4(self.dec_conv4(torch.cat((self.up_conv1(enc4), enc3), dim=1))))
        dec2 = F.relu(self.dec_bn3(self.dec_conv3(torch.cat((self.up_conv0(dec3), enc2), dim=1))))
        dec1 = F.relu(self.dec_bn2(self.dec_conv2(torch.cat((self.up_conv_1(dec2), enc1), dim=1))))
        dec0 = F.relu(self.dec_bn1(self.dec_conv1(torch.cat((self.up_conv_2(dec1), enc0), dim=1))))

        # Final convolution
        return self.final_conv(dec0)
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512, 1024]):
        super(UNet, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder path
        for feature in features:
            self.encoders.append(
                UNet._block(in_channels, feature)
            )
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = UNet._block(features[-1], features[-1] * 2)

        # Decoder path
        for feature in reversed(features):
            self.decoders.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(
                UNet._block(feature * 2, feature)
            )
        
        # Final convolution
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    @staticmethod
    def _block(in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip_connections = []

        # Encoder
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse the skip connections
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.decoders), 2):
            x = self.decoders[idx](x)
            skip_connection = skip_connections[idx // 2]

            # If the input sizes are different, resize the skip connection to match
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoders[idx + 1](x)

        return self.final_layer(x)
    
### CNN-RNN HYBRIDS ###

# SIMPLE CNN-LSTM EXAMPLE #

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,  # for input, forget, cell, and output gates
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along the channel dimension
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
    
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, output_dim):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.output_dim = output_dim

        layers = []
        for i in range(self.num_layers):
            in_channels = self.input_dim if i == 0 else self.hidden_dim
            layers.append(ConvLSTMCell(in_channels, self.hidden_dim, self.kernel_size, bias=True))
        self.layers = nn.ModuleList(layers)
        self.final_conv = nn.Conv2d(self.hidden_dim, self.output_dim, kernel_size=1)

    def forward(self, x, hidden=None):
        # x of shape (batch, seq_len, channels, height, width)
        batch_size, seq_len, _, height, width = x.size()

        if hidden is None:
            hidden = [layer.init_hidden(batch_size, (height, width)) for layer in self.layers]

        current_input = x.transpose(0, 1)  # Swap batch and seq_len dimensions
        for layer_idx, layer in enumerate(self.layers):
            output_inner = []
            for t in range(seq_len):
                hidden[layer_idx] = layer(current_input[t], hidden[layer_idx])
                output_inner.append(hidden[layer_idx][0])
            current_input = torch.stack(output_inner, 0)

        # Take the last output of the sequence
        last_seq_output = current_input[-1]

        # Apply the final convolution to get the output
        output = self.final_conv(last_seq_output)

        return output, hidden

    def init_hidden(self, batch_size, image_size):
        return [layer.init_hidden(batch_size, image_size) for layer in self.layers]
    
# MORE COMPLEX CNN-LSTM EXAMPLE #

class ConvLSTMCellComplex(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, dropout_rate=0.0):
        super(ConvLSTMCellComplex, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.dropout_rate = dropout_rate
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.dropout = nn.Dropout2d(self.dropout_rate)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        c_next = self.dropout(c_next)  # Apply dropout to cell state
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
    
    def init_hidden(self, batch_size, spatial_dims):
        height, width = spatial_dims
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        return (h, c)

class ConvLSTMComplex(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, output_dim, dropout_rate=0.0):
        super(ConvLSTMComplex, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        layers = []
        for i in range(self.num_layers):
            in_channels = self.input_dim if i == 0 else self.hidden_dim
            layers.append(ConvLSTMCellComplex(in_channels, self.hidden_dim, self.kernel_size, bias=True, dropout_rate=self.dropout_rate))
        self.layers = nn.ModuleList(layers)
        self.final_conv = nn.Conv2d(self.hidden_dim, self.output_dim, kernel_size=1)

    def forward(self, x, hidden=None):
        batch_size, seq_len, _, height, width = x.size()
        if hidden is None:
            hidden = [layer.init_hidden(batch_size, (height, width)) for layer in self.layers]
        current_input = x.transpose(0, 1)  # Swap batch and seq_len dimensions
        for layer_idx, layer in enumerate(self.layers):
            output_inner = []
            for t in range(seq_len):
                hidden[layer_idx] = layer(current_input[t], hidden[layer_idx])
                output_inner.append(hidden[layer_idx][0])
            current_input = torch.stack(output_inner, 0)
        last_seq_output = current_input[-1]
        output = self.final_conv(last_seq_output)
        return output, hidden
    

# UNET-LSTM #

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape != skip.shape:
            x = torch.nn.functional.pad(x, (0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]))
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

# class UNetLSTM(nn.Module):
#     def __init__(self, input_channels=3, output_channels=2, hidden_size=512):
#         super().__init__()
#         self.enc1 = EncoderBlock(input_channels, 16)
#         self.enc2 = EncoderBlock(16, 32)
#         self.enc3 = EncoderBlock(32, 64)
#         self.enc4 = EncoderBlock(64, 128)
#         self.enc5 = EncoderBlock(128, 256)
#         self.conv_block = ConvBlock(256, 512)
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         self.lstm = nn.LSTM(512, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 512)
#         self.reshape = nn.Unflatten(1, (512, 1, 1))
#         self.dec5 = DecoderBlock(512, 256)
#         self.dec4 = DecoderBlock(256, 128)
#         self.dec3 = DecoderBlock(128, 64)
#         self.dec2 = DecoderBlock(64, 32)
#         self.dec1 = DecoderBlock(32, 16)
#         self.final_conv = nn.Conv2d(16, output_channels, kernel_size=1)

#     def forward(self, x):
#         batch_size, seq_len, _, _, _ = x.size()
#         outputs = []
#         for t in range(seq_len):
#             x_t, skip1 = self.enc1(x[:, t])
#             x_t, skip2 = self.enc2(x_t)
#             x_t, skip3 = self.enc3(x_t)
#             x_t, skip4 = self.enc4(x_t)
#             x_t, skip5 = self.enc5(x_t)
#             x_t = self.conv_block(x_t)
#             x_t = self.gap(x_t)
#             x_t = x_t.view(batch_size, -1).unsqueeze(1)
#             outputs.append(x_t)
#         x = torch.cat(outputs, dim=1)

#         x, _ = self.lstm(x)
#         x = self.fc(x[:, -1, :])
#         x = self.reshape(x)

#         x = self.dec5(x, skip5)
#         x = self.dec4(x, skip4)
#         x = self.dec3(x, skip3)
#         x = self.dec2(x, skip2)
#         x = self.dec1(x, skip1)
#         x = self.final_conv(x)
#         return x, None

class UNetLSTM(nn.Module):
    def __init__(self, input_channels=3, output_channels=2, hidden_size=512):
        super().__init__()
        self.enc1 = EncoderBlock(input_channels, 16)
        self.enc2 = EncoderBlock(16, 32)
        self.enc3 = EncoderBlock(32, 64)
        self.enc4 = EncoderBlock(64, 128)
        self.enc5 = EncoderBlock(128, 256)
        self.conv_block = ConvBlock(256, 512)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm1 = nn.LSTM(512, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.reshape = nn.Unflatten(1, (512, 1, 1))
        self.dec5 = DecoderBlock(512, 256)
        self.dec4 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128, 64)
        self.dec2 = DecoderBlock(64, 32)
        self.dec1 = DecoderBlock(32, 16)
        self.final_conv = nn.Conv2d(16, output_channels, kernel_size=1)

    def forward(self, x):
        batch_size, seq_len, _, _, _ = x.size()
        outputs = []
        for t in range(seq_len):
            x_t, skip1 = self.enc1(x[:, t])
            x_t, skip2 = self.enc2(x_t)
            x_t, skip3 = self.enc3(x_t)
            x_t, skip4 = self.enc4(x_t)
            x_t, skip5 = self.enc5(x_t)
            x_t = self.conv_block(x_t)
            x_t = self.gap(x_t)
            x_t = x_t.view(batch_size, -1).unsqueeze(1)
            outputs.append(x_t)
        x = torch.cat(outputs, dim=1)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)  # Second LSTM layer output
        x = x[:, -1, :]  # Only the last output from the second LSTM
        x = self.reshape(x)

        x = self.dec5(x, skip5)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        x = self.final_conv(x)
        return x, None