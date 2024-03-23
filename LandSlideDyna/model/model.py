import torch
import torch.nn as nn
import torch.nn.functional as F

#######################################
## TINY MOST BASIC IMPLEMENTATION OF UNET ##
#######################################

class TinyBasicUNet(nn.Module):
    """A basic U-Net architecture for semantic segmentation.

    The U-Net is composed of an encoder (contracting path), a bottleneck, and a decoder (expansive path),
    with skip connections between the encoder and decoder blocks.
    """
    
    def __init__(self, in_channels, out_channels):
        """Initializes the BasicUNet with the given number of input and output channels."""
        super(TinyBasicUNet, self).__init__()

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
        d1 = self.dec1(torch.cat((e4, b), dim=1))
        d2 = self.dec2(torch.cat((e3, d1), dim=1))
        d3 = self.dec3(torch.cat((e2, d2), dim=1))
        d4 = self.dec4(torch.cat((e1, d3), dim=1))

        # Final output
        out = self.out_conv(d4)
        return out


class SmallBasicUNet(nn.Module):
    """A basic U-Net architecture for semantic segmentation.

    The U-Net is composed of an encoder (contracting path), a bottleneck, and a decoder (expansive path),
    with skip connections between the encoder and decoder blocks.
    """
    
    def __init__(self, in_channels, out_channels):
        """Initializes the BasicUNet with the given number of input and output channels."""
        super(SmallBasicUNet, self).__init__()

        # Encoder
        self.enc1 = self.encoder_block(in_channels, 16)
        self.enc2 = self.encoder_block(16, 32)
        self.enc3 = self.encoder_block(32, 64)
        self.enc4 = self.encoder_block(64, 128)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.dec1 = self.decoder_block(256 + 128, 128)  # Adjusted for concatenated channels
        self.dec2 = self.decoder_block(128 + 64, 64)   # Adjusted for concatenated channels
        self.dec3 = self.decoder_block(64 + 32, 32)   # Adjusted for concatenated channels
        self.dec4 = self.decoder_block(32 + 16, 16)     # Adjusted for concatenated channels

        # Final output
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

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
        d1 = self.dec1(torch.cat((e4, b), dim=1))
        d2 = self.dec2(torch.cat((e3, d1), dim=1))
        d3 = self.dec3(torch.cat((e2, d2), dim=1))
        d4 = self.dec4(torch.cat((e1, d3), dim=1))

        # Final output
        out = self.out_conv(d4)
        return out


class MediumBasicUNet(nn.Module):
    """A basic U-Net architecture for semantic segmentation.

    The U-Net is composed of an encoder (contracting path), a bottleneck, and a decoder (expansive path),
    with skip connections between the encoder and decoder blocks.
    """
    
    def __init__(self, in_channels, out_channels):
        """Initializes the BasicUNet with the given number of input and output channels."""
        super(MediumBasicUNet, self).__init__()

        # Encoder
        self.enc1 = self.encoder_block(in_channels, 32)
        self.enc2 = self.encoder_block(32, 64)
        self.enc3 = self.encoder_block(64, 128)
        self.enc4 = self.encoder_block(128, 256)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.dec1 = self.decoder_block(512 + 256, 256)  # Adjusted for concatenated channels
        self.dec2 = self.decoder_block(256 + 128, 128)   # Adjusted for concatenated channels
        self.dec3 = self.decoder_block(128 + 64, 64)   # Adjusted for concatenated channels
        self.dec4 = self.decoder_block(64 + 32, 32)     # Adjusted for concatenated channels

        # Final output
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

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
        d1 = self.dec1(torch.cat((e4, b), dim=1))
        d2 = self.dec2(torch.cat((e3, d1), dim=1))
        d3 = self.dec3(torch.cat((e2, d2), dim=1))
        d4 = self.dec4(torch.cat((e1, d3), dim=1))

        # Final output
        out = self.out_conv(d4)
        return out



################################################
## MEDIUM UNET PLUS: WITH BATCH NORM / DROPOUT ##
################################################

class MediumUNetPlus(nn.Module):
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
        super(MediumUNetPlus, self).__init__()

        # Encoder
        self.enc1 = self.encoder_block(in_channels, 32)
        self.enc2 = self.encoder_block(32, 64)
        self.enc3 = self.encoder_block(64, 128)
        self.enc4 = self.encoder_block(128, 256)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Decoder
        self.dec1 = self.decoder_block(512 + 256, 256, dropout_rate)
        self.dec2 = self.decoder_block(256 + 128, 128, dropout_rate)
        self.dec3 = self.decoder_block(128 + 64, 64, dropout_rate)
        self.dec4 = self.decoder_block(64 + 32, 32, dropout_rate)

        # Final output
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

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

        # Decoder with skip connections
        d1 = self.dec1(torch.cat((e4, b), dim=1))
        d2 = self.dec2(torch.cat((e3, d1), dim=1))
        d3 = self.dec3(torch.cat((e2, d2), dim=1))
        d4 = self.dec4(torch.cat((e1, d3), dim=1))

        # Final output
        out = self.out_conv(d4)

        return out
    





class CNNLSTM(nn.Module):
    def __init__(self, unet, lstm_hidden_size, lstm_layers):
        super(CNNLSTM, self).__init__()
        self.unet = unet
        self.sequence_length = 5  # The length of the input sequences
        
        # Use a dummy input to determine the UNet's output size for one frame
        dummy_input = torch.randn(1, 3, 256, 256)  # Single frame input
        with torch.no_grad():
            dummy_output = self.unet(dummy_input)

        # Flatten the output to get the number of features for the LSTM
        num_features = dummy_output.nelement()

        # Define the LSTM layer with the correct input size
        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True)

        # Define the final fully connected layer
        self.fc = nn.Linear(in_features=lstm_hidden_size,
                            out_features=2 * 256 * 256)  # Output features for each pixel

    def forward(self, x):
        batch_size = x.size(0)
        sequence_output = []

        # Process each frame through UNet and collect outputs
        for t in range(self.sequence_length):
            frame_output = self.unet(x[:, t])
            frame_output = frame_output.view(batch_size, -1)  # Flatten the output
            sequence_output.append(frame_output)

        # Stack the sequence outputs into a batch
        lstm_input = torch.stack(sequence_output, dim=1)

        # Process the sequence through the LSTM
        lstm_output, _ = self.lstm(lstm_input)

        # Process the output of the LSTM with a fully connected layer
        # Assuming we only want the last output of the sequence for prediction
        last_lstm_output = lstm_output[:, -1]
        final_output = self.fc(last_lstm_output)
        final_output = final_output.view(batch_size, 256, 256, 2)  # Reshape to match target

        return final_output
