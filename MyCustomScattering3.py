import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering2D
from pytorch_wavelets import DTCWTForward, ScatLayerj2

import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering2D
from pytorch_wavelets import DTCWTForward, ScatLayer, ScatLayerj2
import os

global_wavelet_scatter = None

def create_wavelet_scatter(J=2, use_pytorch_wavelets=True, in_channels=3):
    global global_wavelet_scatter

    if (global_wavelet_scatter is not None and
        global_wavelet_scatter.J == J and
        global_wavelet_scatter.use_pytorch_wavelets == use_pytorch_wavelets and
        global_wavelet_scatter.in_channels == in_channels):
        print("Returning the existing global_wavelet_scatter object.")
        return global_wavelet_scatter

    # If not, create a new object and assign it to the global variable
    global_wavelet_scatter = WaveletScatter(J=J, use_pytorch_wavelets=use_pytorch_wavelets, in_channels=in_channels)
    print("Created a new global_wavelet_scatter object.")
    return global_wavelet_scatter


class ScatteringGainLayer(nn.Module):
    def __init__(self, num_bands):
        super(ScatteringGainLayer, self).__init__()
        self.gain = nn.Parameter(torch.ones(1, num_bands, 1, 1))

    def forward(self, x):
        return x * self.gain

# class WaveletScatter(nn.Module):
#     def __init__(self, J, use_pytorch_wavelets=True, in_channels=3):
#         super(WaveletScatter, self).__init__()
#         self.J = J
#         self.in_channels = in_channels
#         self.use_pytorch_wavelets = use_pytorch_wavelets
#         if self.use_pytorch_wavelets:
#             self.scatlayer = ScatLayerj2(biort='near_sym_a')
#         self.scattering_cache = {}

#     def get_scattering(self, shape):
#         if shape not in self.scattering_cache:
#             scattering = Scattering2D(J=self.J, shape=shape).cuda()
#             self.scattering_cache[shape] = scattering
#         return self.scattering_cache[shape]

#     def __eq__(self, other):
#         if not isinstance(other, WaveletScatter):
#             return False
#         return (self.J == other.J and
#                 self.use_pytorch_wavelets == other.use_pytorch_wavelets and
#                 self.in_channels == other.in_channels)

#     def forward(self, x):
#         if self.use_pytorch_wavelets:
#             self.scatlayer = self.scatlayer.to(x.device)
#             Y = self.scatlayer(x)
#             # Remove the lowpass outputs and keep only the highpass outputs
#             Y_high = Y[:, 1:, :, :]
#             return Y_high
#         else:
#             batch_size, channels, height, width = x.size()
#             scattering = self.get_scattering((height, width))            
#             scattering_features = scattering(x)

#             scattering_features = scattering_features[:, :, 1:, :, :]

#             num_color_channels = scattering_features.shape[1]
#             scattering_features_reshaped = scattering_features.reshape(
#                 1, 
#                 num_color_channels * scattering_features.shape[2], 
#                 scattering_features.shape[3], 
#                 scattering_features.shape[4]
#             )

#             return scattering_features_reshaped
   
class WaveletScatter(nn.Module):
    def __init__(self, J, use_pytorch_wavelets=True, in_channels=3):
        super(WaveletScatter, self).__init__()
        self.J = J
        self.in_channels = in_channels
        self.use_pytorch_wavelets = use_pytorch_wavelets
        self.scattering_cache = {}

        if self.use_pytorch_wavelets:
            self.scatlayer = ScatLayer(biort='near_sym_a', combine_colour=True)
            self.expected_out_channels = 6
        else:
            self.expected_out_channels = 80 * in_channels  # Estimate for J=2

        # Initialize gain layer after scattering
        self.gain_layer = ScatteringGainLayer(num_bands=self.expected_out_channels)

    def get_scattering(self, shape):
        if shape not in self.scattering_cache:
            scattering = Scattering2D(J=self.J, shape=shape).cuda()
            self.scattering_cache[shape] = scattering
        return self.scattering_cache[shape]

    def __eq__(self, other):
        if not isinstance(other, WaveletScatter):
            return False
        return (self.J == other.J and
                self.use_pytorch_wavelets == other.use_pytorch_wavelets and
                self.in_channels == other.in_channels)

    def forward(self, x):
        if self.use_pytorch_wavelets:
            self.scatlayer = self.scatlayer.to(x.device)
            Y = self.scatlayer(x)
            Y_high = Y[:, 3:, :, :]
            Y_high = self.gain_layer.to(x.device)(Y_high)  # Apply learnable gain
            return Y_high
        else:
            batch_size, channels, height, width = x.size()
            scattering = self.get_scattering((height, width)).to(x.device)
            scattering_features = scattering(x)  # [B, C, Bands, H, W]
            scattering_features = scattering_features[:, :, 1:, :, :]
            num_color_channels = scattering_features.shape[1]
            reshaped = scattering_features.reshape(
                batch_size,
                num_color_channels * scattering_features.shape[2],
                scattering_features.shape[3],
                scattering_features.shape[4]
            )
            reshaped = self.gain_layer(reshaped)  # Apply learnable gain
            return reshaped


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.local_att = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.global_att = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv = nn.Conv2d(inter_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        batch, channel, height, width = x.size()

        local_feats = self.local_att(x)
        global_feats = self.global_att(x).mean(dim=(2, 3), keepdim=True)
        combined_feats = torch.sigmoid(local_feats + global_feats)
        combined_feats = self.conv(combined_feats)  # Adjust to match the number of input channels
        output = combined_feats * x
        return output

# Define the Fusion Module for correlating scattering coefficient channels
class FusionModule(nn.Module):
    def __init__(self, in_channels, fusion_channels):
        super(FusionModule, self).__init__()
        self.in_channels = in_channels
        self.fusion_channels = fusion_channels

        # Ensure fusion_channels is divisible by in_channels or adjust the groups
        # Using depthwise convolution followed by pointwise to achieve grouped effect
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)  # Depthwise convolution
        self.pointwise_conv = nn.Conv2d(in_channels, fusion_channels, kernel_size=1)  # Pointwise convolution for channel mixing
        
        # Channel Attention mechanism to enhance the most relevant features
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fusion_channels, fusion_channels // 4, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(fusion_channels // 4, fusion_channels, kernel_size=1),
            nn.Softplus()
        )
        
        # Final pointwise convolution for further refinement
        self.final_conv = nn.Conv2d(fusion_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # Apply depthwise convolution followed by pointwise
        x = self.depthwise_conv(x)
        fused = self.pointwise_conv(x)
        
        # Apply channel attention
        attention_weights = self.channel_att(fused)
        fused = fused * attention_weights
        
        # Further refine and mix the channels
        fused = self.final_conv(fused)
        
        return fused + x  # Skip connection to retain original features


# Define the Global Attention Module to capture global context
class GlobalAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(GlobalAttentionModule, self).__init__()
        self.in_channels = in_channels
        
        # Spatial attention pooling to capture global spatial information
        self.spatial_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Softplus()
        )
        
        # Channel attention to refine global context
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Softplus()
        )

    def forward(self, x):
        # Apply spatial attention pooling
        spatial_weights = self.spatial_att(x)
        spatial_out = x * spatial_weights
        
        # Apply channel attention
        channel_weights = self.channel_att(spatial_out)
        out = spatial_out * channel_weights
        
        return out + x  # Skip connection to enhance original features

class EnhancedWaveletScatteringUNetPlusPlusWithLeaky(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, J=2, use_pytorch_wavelets=True):
        super(EnhancedWaveletScatteringUNetPlusPlusWithLeaky, self).__init__()
        self.use_pytorch_wavelets = use_pytorch_wavelets
        self.in_channels = in_channels
        self.J = J

        # Set scattering channels based on the chosen method
        if use_pytorch_wavelets:
            scattering_channels = 12 #146 if in_channels == 3 else 48  # For DTCWT with J=2
        else:
            scattering_channels = 80 * in_channels  # For kymatio with J=2

        global global_wavelet_scatter
        global_wavelet_scatter = WaveletScatter(J=J, use_pytorch_wavelets=use_pytorch_wavelets, in_channels=in_channels)

        
        self.fusion_module = FusionModule(scattering_channels, fusion_channels=64)
        
        self.global_attention = GlobalAttentionModule(scattering_channels)

        self.conv0_0 = self._conv_block(scattering_channels, 64)
        self.conv1_0 = self._conv_block(64, 128)
        self.conv2_0 = self._conv_block(128, 256)
        self.conv3_0 = self._conv_block(256, 512)
        self.conv4_0 = self._conv_block(512, 1024)

        self.attention0 = AttentionBlock(64, 32)
        self.attention1 = AttentionBlock(128, 64)
        self.attention2 = AttentionBlock(256, 128)
        self.attention3 = AttentionBlock(512, 256)
        self.attention4 = AttentionBlock(1024, 512)

        self.deep_supervision1 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.deep_supervision2 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.deep_supervision3 = nn.Conv2d(256, out_channels, kernel_size=1)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),  # Group Normalization
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.2),  # Dropout for regularization
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, mask=None):
        original_size = x.size()[2:]  # Get the original size (H, W)
        scattering_features = x
        # global global_wavelet_scatter
        # if global_wavelet_scatter is None:
        #     global_wavelet_scatter = create_wavelet_scatter(J=self.J, use_pytorch_wavelets=self.use_pytorch_wavelets, in_channels=self.in_channels)
        # scattering_features = global_wavelet_scatter(x)
        
        if mask is not None:
            resized_mask = F.interpolate(mask, size=scattering_features.shape[2:], mode='nearest')
            scattering_features = scattering_features * resized_mask

        scattering_features = self.fusion_module(scattering_features)
        
        scattering_features = self.global_attention(scattering_features)

        # Encoder pathway
        x0_0 = self.conv0_0(scattering_features)
        x0_0 = self.attention0(x0_0)
        x1_0 = self.conv1_0(F.max_pool2d(x0_0, 2))
        x1_0 = self.attention1(x1_0)
        x2_0 = self.conv2_0(F.max_pool2d(x1_0, 2))
        x2_0 = self.attention2(x2_0)
        x3_0 = self.conv3_0(F.max_pool2d(x2_0, 2))
        x3_0 = self.attention3(x3_0)
        x4_0 = self.conv4_0(F.max_pool2d(x3_0, 2))
        x4_0 = self.attention4(x4_0)

        # Decoder pathway with deep supervision
        ds1 = self.deep_supervision1(F.interpolate(x0_0, scale_factor=2, mode='bilinear', align_corners=True))
        ds2 = self.deep_supervision2(F.interpolate(x1_0, scale_factor=4, mode='bilinear', align_corners=True))
        ds3 = self.deep_supervision3(F.interpolate(x2_0, scale_factor=8, mode='bilinear', align_corners=True))

        # Combine deep supervision outputs
        ds_combined = (ds1 + ds2 + ds3) / 3

        # Final output
        output = self.final(x0_0)
        output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=True)  # Upsample to match the input size

        if output.size() != ds_combined.size():
                ds_combined = F.interpolate(ds_combined, size=output.size()[2:], mode='bilinear', align_corners=True)

        return  (output + ds_combined) * mask if mask is not None else (output + ds_combined)
