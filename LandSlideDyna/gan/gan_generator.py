import torch
import torch.nn as nn
import torch.nn.functional as F

# THESE TWO WERE AN ATTEMPT TO ADD ATTENTION

# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()
#         self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         attention_map = self.conv(x)
#         attention_map = self.softmax(attention_map)
#         attention_map = attention_map.expand_as(x)
#         x = x * attention_map
#         return x
    
# class UNetUp(nn.Module):
#     def __init__(self, in_size, out_size, kernel_size, dropout=False):
#         super(UNetUp, self).__init__()
#         self.conv = nn.ConvTranspose2d(in_size, out_size, kernel_size, stride=2, padding=1, bias=False)
#         self.bn = nn.BatchNorm2d(out_size)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(0.5) if dropout else None
#         self.attention = SelfAttention(out_size * 2)

#     def forward(self, x, skip_input):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         if self.dropout:
#             x = self.dropout(x)
#         x = torch.cat((x, skip_input), 1)
#         x = self.attention(x)
#         return x

# THESE TRY TO ADD SOME COMPLEXITY TO THE UP and DOWN LAYERS

# class UNetDown(nn.Module):
#     def __init__(self, in_size, out_size, apply_batchnorm=True):
#         super(UNetDown, self).__init__()
#         layers = [
#             nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(out_size, out_size, kernel_size=3, stride=2, padding=1, bias=False),
#         ]
#         if apply_batchnorm:
#             layers.insert(2, nn.BatchNorm2d(out_size))
#             layers.insert(5, nn.BatchNorm2d(out_size))
#         layers.append(nn.LeakyReLU(0.2))

#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)

# class UNetUp(nn.Module):
#     def __init__(self, in_size, out_size, dropout=False):
#         super(UNetUp, self).__init__()
#         self.conv1 = nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_size)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_size)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_size)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(0.5) if dropout else None

#     def forward(self, x, skip_input):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = torch.cat((x, skip_input), 1)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu3(x)
#         if self.dropout:
#             x = self.dropout(x)
#         return x

# class GeneratorUNet(nn.Module):
#     def __init__(self, input_channels=2, output_channels=2, kernel_size=4, apply_mask=False):
#         super(GeneratorUNet, self).__init__()

#         self.down1 = UNetDown(input_channels, 64)
#         self.down2 = UNetDown(64, 128)
#         self.down3 = UNetDown(128, 256)
#         self.down4 = UNetDown(256, 512)
#         self.down5 = UNetDown(512, 512)
#         self.down6 = UNetDown(512, 512)
#         self.down7 = UNetDown(512, 512)

#         self.middle = nn.Sequential(
#             nn.Conv2d(512, 512, stride=2, padding=1, kernel_size=kernel_size),
#             nn.ReLU(inplace=True)
#         )

#         self.up1 = UNetUp(512, 512, dropout=True)
#         self.up2 = UNetUp(1024, 512, dropout=True)
#         self.up3 = UNetUp(1024, 512, dropout=True)
#         self.up4 = UNetUp(1024, 512)
#         self.up5 = UNetUp(1024, 256)
#         self.up6 = UNetUp(512, 128)
#         self.up7 = UNetUp(256, 64)

#         self.final = nn.Sequential(
#             nn.ConvTranspose2d(128, output_channels, stride=2, kernel_size=4, padding=1),
#             nn.Tanh()
#         )

#         self.apply_mask = apply_mask

#     def forward(self, x, mask=None):
#         # Encoder
#         d1 = self.down1(x)
#         d2 = self.down2(d1)
#         d3 = self.down3(d2)
#         d4 = self.down4(d3)
#         d5 = self.down5(d4)
#         d6 = self.down6(d5)
#         d7 = self.down7(d6)

#         # Middle
#         middle = self.middle(d7)

#         # Decoder with skip connections
#         u1 = self.up1(middle, d7)
#         u2 = self.up2(u1, d6)
#         u3 = self.up3(u2, d5)
#         u4 = self.up4(u3, d4)
#         u5 = self.up5(u4, d3)
#         u6 = self.up6(u5, d2)
#         u7 = self.up7(u6, d1)

#         # Final layer
#         result = self.final(u7)


#         # Apply the mask if enabled
    
#         if self.apply_mask and mask is not None:

#             if mask.dim() == 3:
#                 mask = mask.unsqueeze(1)  # Add channel dimension if missing
#                 mask = mask.repeat(1, result.shape[1], 1, 1)  # Duplicate along channel dimension
#             elif mask.dim() == 4:
#                 mask = mask.repeat(1, result.shape[1], 1, 1)  # Duplicate along channel dimension
#             else:
#                 raise ValueError(f"Unexpected mask dimension: {mask.dim()}")
            

#             result = result * mask

#         return result





# THESE WORK SORT OF

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, apply_batchnorm=True):
        super(UNetDown, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, kernel_size, stride=2, padding=1, bias=False)
        ]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, dropout=False):
        super(UNetUp, self).__init__()
        self.conv = nn.ConvTranspose2d(in_size, out_size, kernel_size, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5) if dropout else None

    def forward(self, x, skip_input):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, input_channels=2, output_channels=2, kernel_size=2, apply_mask=False):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(input_channels, 64, kernel_size, apply_batchnorm=False)
        self.down2 = UNetDown(64, 128, kernel_size)
        self.down3 = UNetDown(128, 256, kernel_size)
        self.down4 = UNetDown(256, 512, kernel_size)
        self.down5 = UNetDown(512, 512, kernel_size)
        self.down6 = UNetDown(512, 512, kernel_size)
        self.down7 = UNetDown(512, 512, kernel_size)

        self.middle = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = UNetUp(512, 512, kernel_size, dropout=True)
        self.up2 = UNetUp(1024, 512, kernel_size, dropout=True)
        self.up3 = UNetUp(1024, 512, kernel_size, dropout=True)
        self.up4 = UNetUp(1024, 512, kernel_size)
        self.up5 = UNetUp(1024, 256, kernel_size)
        self.up6 = UNetUp(512, 128, kernel_size)
        self.up7 = UNetUp(256, 64, kernel_size)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, output_channels, kernel_size, stride=2, padding=1),
            nn.Tanh()
        )

        self.apply_mask = apply_mask

    def forward(self, x, mask=None):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        # Middle
        middle = self.middle(d7)

        # Decoder with skip connections
        u1 = self.up1(middle, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        # Final layer
        result = self.final(u7)


        # Apply the mask if enabled
    
        if self.apply_mask and mask is not None:

            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Add channel dimension if missing
                mask = mask.repeat(1, result.shape[1], 1, 1)  # Duplicate along channel dimension
            elif mask.dim() == 4:
                mask = mask.repeat(1, result.shape[1], 1, 1)  # Duplicate along channel dimension
            else:
                raise ValueError(f"Unexpected mask dimension: {mask.dim()}")
            

            result = result * mask

        return result

