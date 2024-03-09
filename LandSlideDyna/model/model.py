import torch
import torch.nn as nn
import torch.nn.functional as F

#######################################
## SMALL MOST BASIC IMPLEMENTATION OF UNET ##
#######################################

class SmallBasicUNet(nn.Module):
    """A basic U-Net architecture for semantic segmentation.

    The U-Net is composed of an encoder (contracting path), a bottleneck, and a decoder (expansive path),
    with skip connections between the encoder and decoder blocks.
    """
    
    def __init__(self, in_channels, out_channels):
        """Initializes the BasicUNet with the given number of input and output channels."""
        super(SmallBasicUNet, self).__init__()

        # Encoder
        self.enc1 = self.encoder_block(in_channels, 8)
        self.enc2 = self.encoder_block(8, 16)
        self.enc3 = self.encoder_block(16, 32)
        self.enc4 = self.encoder_block(32, 64)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.dec1 = self.decoder_block(128 + 64, 64)  # Adjusted for concatenated channels
        self.dec2 = self.decoder_block(64 + 32, 32)   # Adjusted for concatenated channels
        self.dec3 = self.decoder_block(32 + 16, 16)   # Adjusted for concatenated channels
        self.dec4 = self.decoder_block(16 + 8, 8)     # Adjusted for concatenated channels

        # Final output
        self.out_conv = nn.Conv2d(8, out_channels, kernel_size=1)

    def encoder_block(self, in_channels, out_channels):
        """Defines an encoder block with Convolution, ReLU activation, and MaxPooling."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def decoder_block(self, in_channels, out_channels):
        """Defines a decoder block with Convolution, ReLU activation, and Upsampling."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections and cropping
        d1 = self.dec1(torch.cat((self.crop(e4, b), b), dim=1))
        d2 = self.dec2(torch.cat((self.crop(e3, d1), d1), dim=1))
        d3 = self.dec3(torch.cat((self.crop(e2, d2), d2), dim=1))
        d4 = self.dec4(torch.cat((self.crop(e1, d3), d3), dim=1))

        # Final output
        out = self.out_conv(d4)
        return out

    @staticmethod
    def crop(encoder_layer, decoder_layer):
        """Crop the encoder_layer to the size of the decoder_layer."""
        if encoder_layer.size()[2:] != decoder_layer.size()[2:]:
            ds_height = decoder_layer.size(2)
            ds_width = decoder_layer.size(3)
            encoder_layer = F.interpolate(encoder_layer, size=(ds_height, ds_width), mode='nearest')
        return encoder_layer

# class SmallBasicUNet(nn.Module):
#     """A basic U-Net architecture for semantic segmentation.

#     The U-Net is composed of an encoder (contracting path), a bottleneck, and a decoder (expansive path),
#     with skip connections between the encoder and decoder blocks.

#     Attributes:
#         enc1: First encoder block.
#         enc2: Second encoder block.
#         enc3: Third encoder block.
#         enc4: Fourth encoder block.
#         bottleneck: The bottleneck part of the network.
#         dec1: First decoder block.
#         dec2: Second decoder block.
#         dec3: Third decoder block.
#         dec4: Fourth decoder block.
#         out_conv: Final output convolutional layer.
#     """

#     def __init__(self, in_channels, out_channels):
#         """Initializes the BasicUNet with the given number of input and output channels.

#         Args:
#             in_channels: The number of input channels.
#             out_channels: The number of output channels.
#         """
#         super(SmallBasicUNet, self).__init__()

#         # Encoder
#         self.enc1 = self.encoder_block(in_channels, 8)
#         self.enc2 = self.encoder_block(8, 16)
#         self.enc3 = self.encoder_block(16, 32)
#         self.enc4 = self.encoder_block(32, 64)

#         # Bottleneck
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU()
#         )

#         # Decoder
#         self.dec1 = self.decoder_block(128, 64)
#         self.dec2 = self.decoder_block(64, 32)
#         self.dec3 = self.decoder_block(32, 16)
#         self.dec4 = self.decoder_block(16, 8)

#         # Final output
#         self.out_conv = nn.Conv2d(8, out_channels, kernel_size=1)

#     def encoder_block(self, in_channels, out_channels):
#         """Defines an encoder block with Convolution, ReLU activation, and MaxPooling.

#         Args:
#             in_channels: The number of input channels for the block.
#             out_channels: The number of output channels for the block.

#         Returns:
#             An nn.Sequential module comprising the encoder block layers.
#         """
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#     def decoder_block(self, in_channels, out_channels):
#         """Defines a decoder block with Convolution, ReLU activation, and Upsampling.

#         Args:
#             in_channels: The number of input channels for the block.
#             out_channels: The number of output channels for the block.

#         Returns:
#             An nn.Sequential module comprising the decoder block layers.
#         """
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )

#     def forward(self, x):
#         """Defines the forward pass of the BasicUNet.

#         Args:
#             x: The input tensor.

#         Returns:
#             The output tensor after passing through the U-Net.
#         """
#         # Encoder
#         e1 = self.enc1(x)
#         e2 = self.enc2(e1)
#         e3 = self.enc3(e2)
#         e4 = self.enc4(e3)

#         # Bottleneck
#         b = self.bottleneck(e4)

#         # Decoder
#         d1 = self.dec1(b)
#         d1 = torch.cat((d1, e4), dim=1)  # Corrected: Concatenate after each decoder block
        
#         d2 = self.dec2(d1)
#         d2 = torch.cat((d2, e3), dim=1)  # Corrected: Concatenate after each decoder block
        
#         d3 = self.dec3(d2)
#         d3 = torch.cat((d3, e2), dim=1)  # Corrected: Concatenate after each decoder block
        
#         d4 = self.dec4(d3)
#         d4 = torch.cat((d4, e1), dim=1)  # Corrected: Concatenate after each decoder block
        
#         out = self.out_conv(d4)

#         return out




#######################################
## MOST BASIC IMPLEMENTATION OF UNET ##
#######################################

class BasicUNet(nn.Module):
    """A basic U-Net architecture for semantic segmentation.

    The U-Net is composed of an encoder (contracting path), a bottleneck, and a decoder (expansive path),
    with skip connections between the encoder and decoder blocks.

    Attributes:
        enc1: First encoder block.
        enc2: Second encoder block.
        enc3: Third encoder block.
        enc4: Fourth encoder block.
        bottleneck: The bottleneck part of the network.
        dec1: First decoder block.
        dec2: Second decoder block.
        dec3: Third decoder block.
        dec4: Fourth decoder block.
        out_conv: Final output convolutional layer.
    """

    def __init__(self, in_channels, out_channels):
        """Initializes the BasicUNet with the given number of input and output channels.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
        """
        super(BasicUNet, self).__init__()

        # Encoder
        self.enc1 = self.encoder_block(in_channels, 64)
        self.enc2 = self.encoder_block(64, 128)
        self.enc3 = self.encoder_block(128, 256)
        self.enc4 = self.encoder_block(256, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.dec1 = self.decoder_block(1024, 512)
        self.dec2 = self.decoder_block(512, 256)
        self.dec3 = self.decoder_block(256, 128)
        self.dec4 = self.decoder_block(128, 64)

        # Final output
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def encoder_block(self, in_channels, out_channels):
        """Defines an encoder block with Convolution, ReLU activation, and MaxPooling.

        Args:
            in_channels: The number of input channels for the block.
            out_channels: The number of output channels for the block.

        Returns:
            An nn.Sequential module comprising the encoder block layers.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def decoder_block(self, in_channels, out_channels):
        """Defines a decoder block with Convolution, ReLU activation, and Upsampling.

        Args:
            in_channels: The number of input channels for the block.
            out_channels: The number of output channels for the block.

        Returns:
            An nn.Sequential module comprising the decoder block layers.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        """Defines the forward pass of the BasicUNet.

        Args:
            x: The input tensor.

        Returns:
            The output tensor after passing through the U-Net.
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d1 = self.dec1(b)
        d1 = torch.cat((d1, e4), dim=1)  # Corrected: Concatenate after each decoder block
        
        d2 = self.dec2(d1)
        d2 = torch.cat((d2, e3), dim=1)  # Corrected: Concatenate after each decoder block
        
        d3 = self.dec3(d2)
        d3 = torch.cat((d3, e2), dim=1)  # Corrected: Concatenate after each decoder block
        
        d4 = self.dec4(d3)
        d4 = torch.cat((d4, e1), dim=1)  # Corrected: Concatenate after each decoder block
        
        out = self.out_conv(d4)

        return out


################################################
## BASIC UNET PLUS: WITH BATCH NORM / DROPOUT ##
################################################

class BasicUNetPLUS(nn.Module):
    """A basic U-Net architecture for semantic segmentation with dropout regularization.

    Attributes:
        enc1: First encoder block.
        enc2: Second encoder block.
        enc3: Third encoder block.
        enc4: Fourth encoder block.
        bottleneck: The bottleneck part of the network including dropout layers.
        dec1: First decoder block.
        dec2: Second decoder block.
        dec3: Third decoder block.
        dec4: Fourth decoder block.
        out_conv: Final output convolutional layer.
    """

    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        """Initializes the BasicUNet with the given number of input and output channels and dropout rate.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            dropout_rate: The dropout rate to use in the bottleneck and decoder blocks.
        """
        super(BasicUNet, self).__init__()

        # Encoder
        self.enc1 = self.encoder_block(in_channels, 64)
        self.enc2 = self.encoder_block(64, 128)
        self.enc3 = self.encoder_block(128, 256)
        self.enc4 = self.encoder_block(256, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Decoder
        self.dec1 = self.decoder_block(1024, 512, dropout_rate)
        self.dec2 = self.decoder_block(512, 256, dropout_rate)
        self.dec3 = self.decoder_block(256, 128, dropout_rate)
        self.dec4 = self.decoder_block(128, 64, dropout_rate)

        # Final output
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def encoder_block(self, in_channels, out_channels):
        """Creates an encoder block with Convolution, Batch Normalization, ReLU activation, and MaxPooling.

        Args:
            in_channels: The number of input channels for the block.
            out_channels: The number of output channels for the block.

        Returns:
            An nn.Sequential module comprising the encoder block layers.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def decoder_block(self, in_channels, out_channels, dropout_rate):
        """Creates a decoder block with Convolution, Batch Normalization, ReLU activation, Dropout, and Upsampling.

        Args:
            in_channels: The number of input channels for the block.
            out_channels: The number of output channels for the block.
            dropout_rate: The dropout rate to use after convolutional layers.

        Returns:
            An nn.Sequential module comprising the decoder block layers.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        """Defines the forward pass of the BasicUNet with skip connections.

        Args:
            x: The input tensor.

        Returns:
            The output tensor after passing through the U-Net.
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d1 = self.dec1(b)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        d4 = self.dec4(d3)

        # Skip connections
        d1 = torch.cat((d1, e4), dim=1)
        d2 = torch.cat((d2, e3), dim=1)
        d3 = torch.cat((d3,e2), dim=1)
        d4 = torch.cat((d4, e1), dim=1)

        # Final output
        out = self.out_conv(d4)

        return out
    


#####################################
## VARIABLE UNET - SIZE ADJUSTABLE ##
#####################################

class UNetBase(nn.Module):
    """Base class for constructing U-Net architecture with encoder and decoder blocks.

    This class provides the basic building blocks for the U-Net architecture but does not implement
    the full network. It should be extended to create a complete U-Net model with specific configurations.
    """

    def __init__(self):
        super(UNetBase, self).__init__()

    def encoder_block(self, in_channels, out_channels, use_batchnorm=True):
        """Creates an encoder block for the U-Net architecture.

        Args:
            in_channels: The number of input channels for the block.
            out_channels: The number of output channels for the block.
            use_batchnorm: Whether to use batch normalization after the convolutional layer.

        Returns:
            A sequential container of layers constituting the encoder block.
        """
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def decoder_block(self, in_channels, out_channels, use_batchnorm=True):
        """Creates a decoder block for the U-Net architecture.

        Args:
            in_channels: The number of input channels for the block.
            out_channels: The number of output channels for the block.
            use_batchnorm: Whether to use batch normalization after the convolutional layer.

        Returns:
            A sequential container of layers constituting the decoder block.
        """
        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)


class UNet(UNetBase):
    """Implementation of the U-Net architecture with customizable parameters.

    Attributes:
        enc1: First encoder block.
        enc2: Second encoder block.
        enc3: Third encoder block.
        enc4: Fourth encoder block.
        bottleneck: Bottleneck block with dropout layers.
        dec1: First decoder block.
        dec2: Second decoder block.
        dec3: Third decoder block.
        dec4: Fourth decoder block.
        out_conv: Final convolutional layer to produce the output.
    """

    def __init__(self, in_channels, out_channels, base_filters=64, use_batchnorm=True, dropout_rate=0.5):
        """Initializes the U-Net model with the specified parameters.

        Args:
            in_channels: The number of input channels for the network.
            out_channels: The number of output channels for the network.
            base_filters: The number of filters for the first encoder block.
            use_batchnorm: Whether to use batch normalization.
            dropout_rate: The dropout rate to be used in the bottleneck block.
        """
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.encoder_block(in_channels, base_filters, use_batchnorm)
        self.enc2 = self.encoder_block(base_filters, base_filters*2, use_batchnorm)
        self.enc3 = self.encoder_block(base_filters*2, base_filters*4, use_batchnorm)
        self.enc4 = self.encoder_block(base_filters*4, base_filters*8, use_batchnorm)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters*8, base_filters*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*16) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(base_filters*16, base_filters*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*16) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Decoder
        self.dec1 = self.decoder_block(base_filters*16, base_filters*8, use_batchnorm)
        self.dec2 = self.decoder_block(base_filters*8, base_filters*4, use_batchnorm)
        self.dec3 = self.decoder_block(base_filters*4, base_filters*2, use_batchnorm)
        self.dec4 = self.decoder_block(base_filters*2, base_filters, use_batchnorm)

        # Final output
        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        """Defines the forward pass of the U-Net model.

        Args:
            x: The input tensor to the U-Net model.

        Returns:
            The output tensor of the U-Net model after applying all blocks and layers.
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d1 = self.dec1(b)
        d1 = torch.cat((d1, e4), dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat((d2, e3), dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat((d3, e2), dim=1)
        d4 = self.dec4(d3)
        d4 = torch.cat((d4, e1), dim=1)

        # Final output
        out = self.out_conv(d4)
        return out



class CNNLSTM(nn.Module):
    def __init__(self, in_channels, lstm_hidden_size, lstm_layers=1, n_classes=1, image_size=(512, 512)):
        super(CNNLSTM, self).__init__()
        self.unet = SmallBasicUNet(in_channels=in_channels, out_channels=n_classes)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.n_classes = n_classes

        # Calculate the number of features for LSTM input
        self.num_features = image_size[0] * image_size[1] * n_classes
        self.lstm = nn.LSTM(input_size=self.num_features,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True)
        self.final_fc = nn.Linear(lstm_hidden_size, n_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        # Process each time step through U-Net
        c_out = []
        for t in range(timesteps):
            c_out_t = self.unet(x[:, t])
            c_out.append(c_out_t)
        # Stack the outputs for each time step
        c_out = torch.stack(c_out, dim=1)
        # Flatten the U-Net output for the LSTM, preserving the temporal sequence
        r_out = c_out.view(batch_size, timesteps, -1)

        # Optimize memory layout of LSTM weights
        self.lstm.flatten_parameters()

        # Pass features through LSTM
        lstm_out, (h_n, c_n) = self.lstm(r_out)
        # Take only the output of the last LSTM cell
        final_output = self.final_fc(lstm_out[:, -1])
        return final_output

# Example usage
# model = CNNLSTM(in_channels=3, lstm_hidden_size=256, lstm_layers=2, n_classes=1, image_size=(512, 512))
# class CNNLSTM(nn.Module):
#     """CNN-LSTM model for learning the sequence of landslide debris travel."""
#     def __init__(self, in_channels, lstm_hidden_size, lstm_layers=1, n_classes=1):
#         super(CNNLSTM, self).__init__()
#         self.unet = BasicUNet(n_channels=in_channels, n_classes=n_classes)
#         self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)
#         self.final_fc = nn.Linear(lstm_hidden_size, n_classes)

#     def forward(self, x):
#         batch_size, timesteps, C, H, W = x.size()
#         # Flatten the timesteps into the batch dimension
#         c_in = x.view(batch_size * timesteps, C, H, W)
#         c_out = self.unet(c_in)
#         # Reconstruct the temporal dimension
#         c_out = c_out.view(batch_size, timesteps, -1)

#         # Pass features through LSTM
#         lstm_out, (h_n, c_n) = self.lstm(c_out)
#         # Take only the output of the last LSTM cell
#         final_output = self.final_fc(lstm_out[:, -1])
#         return final_output