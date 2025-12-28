from typing import Optional, Union, List
import torch
import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
import segmentation_models_pytorch.base.initialization as init

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
import segmentation_models_pytorch.base.initialization as init
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union, List
import torch
import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
import segmentation_models_pytorch.base.initialization as init

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.unetplusplus.decoder import UnetPlusPlusDecoder

import segmentation_models_pytorch.base.initialization as init
import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.unetplusplus.decoder  import DecoderBlock, CenterBlock
from segmentation_models_pytorch.manet.decoder import MFAB, PAB

class MyMAnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        reduction=16,
        use_batchnorm=True,
        pab_channels=64,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]

        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        self.center = PAB(head_channels, head_channels, pab_channels=pab_channels)

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm)  # no attention type here
        blocks = [
            MFAB(in_ch, skip_ch, out_ch, reduction=reduction, **kwargs)
            if skip_ch > 0
            else DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        # for the last we dont have skip connection -> use simple decoder block
        self.blocks = nn.ModuleList(blocks)

    def forward(self, inputFeatures):
        #inputFeatures = [inputFeatures] - used only for displaying nicely the architecture

        dense_xs = []
        for i in range(len(inputFeatures)):

            features = inputFeatures[i][1:]  # remove first skip with same spatial resolution
            features = features[::-1]  # reverse channels to start from head of encoder

            head = features[0]
            skips = features[1:]

            x = self.center(head)
            for i, decoder_block in enumerate(self.blocks):
                skip = skips[i] if i < len(skips) else None
                x = decoder_block(x, skip)
            dense_xs.append(x)

        return dense_xs

class MyUnetPlusPlusDecoderWithGlobalAverage(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
        numberofEncoders = 1,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx] 
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1] 
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], **kwargs
        )
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, inputFeatures):
            # inputFeatures = [inputFeatures] - used only for displaying nicely the architecture
            dense_xs = []
            featuresX = inputFeatures
            results = []


            for i in range(len(featuresX[0])): # ..1..6
                temp = []
                
                for j in range(0, len(featuresX)):
                    temp.append(featuresX[j][i])
                temp = torch.cat(temp, dim=0)
                results.append(temp)

            features = results[1:]  # remove first skip with same spatial resolution
            features = features[::-1]  # reverse channels to start from head of encoder
            # start building dense connections
            dense_x = {}
            for layer_idx in range(len(self.in_channels) - 1):
                for depth_idx in range(self.depth - layer_idx):
                    if layer_idx == 0:
                        output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
                        dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                    else:
                        dense_l_i = depth_idx + layer_idx
                        cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
                        cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                        dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
                            dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features
                        )
            dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth-1}"])
            return dense_x[f"x_{0}_{self.depth}"]
    
class MyUnetPlusPlusWithGlobalAverage(torch.nn.Module):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        numberofEncoders: int = 1,
        device: str = "cpu",
        activation: Optional[Union[str, callable]] = None,
        decoderType : str = "unetplusplus",# unetplusplus or manet 
    ):
        super().__init__()

        if encoder_name.startswith("mit_b"):
            raise ValueError("UnetPlusPlus is not support encoder_name={}".format(encoder_name))

        self.encoders = []
        for i in range(numberofEncoders):
            self.encoders.append(get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=None,
            ).to(device))
        
        if decoderType == "unetplusplus":
            self.decoder = MyUnetPlusPlusDecoderWithGlobalAverage(
                numberofEncoders = numberofEncoders,
                encoder_channels=self.encoders[0].out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                center=True if encoder_name.startswith("vgg") else False,
                attention_type=decoder_attention_type,
            )
        else:
            self.decoder = MyMAnetDecoder(
                encoder_channels=self.encoders[0].out_channels,
                decoder_channels=decoder_channels
            )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        # self.segmentation_head_high_res = SegmentationHead(
        #     in_channels=decoder_channels[-1] * numberofEncoders,
        #     out_channels=192,  # Adjusted to match the decoder's output channels
        #     activation=activation,
        #     kernel_size=3,
        # )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_global = nn.Linear(192, 1)  # Input size adjusted to match the output of global_avg
        self.fc_merge = nn.Conv2d(193, classes, kernel_size=3, padding=1)  # Merge 192 high-res + 1 global context

        self.name = f"MyunetplusplusWithGlobalAverage-{encoder_name} - {decoderType}"
        print(self.name)
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

        #init.initialize_head(self.segmentation_head_high_res)
        #init.initialize_head(self.fc_merge)

        #self.segmentation_head_high_res_dropout = nn.Dropout(p=0.4)  # High dropout for segmentation head
        #self.fc_merge_dropout = nn.Dropout(p=0.3)  # Dropout for final merged layer        

    def forward(self, x):
        x_split = torch.split(x, 1, dim=1)   

        features = []
        for i, encoder in enumerate(self.encoders):
            features.append(encoder(x_split[i].float()))

        features_concat = features

        # Decode the features
        decoder_output = self.decoder(features_concat)
        masks = self.segmentation_head(decoder_output)

        H = masks.size(2)
        W = masks.size(3)
        
        masks = masks.permute(2, 3, 0, 1)
        global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  
        masks = global_avg_pool(masks)
        masks = masks.permute(2, 3, 0, 1)
        return masks

        # # Concatenate the decoder outputs (assuming they are along the channel dimension)
        # if isinstance(decoder_output, list):
        #     decoder_output = torch.cat(decoder_output, dim=1)

        # # High resolution segmentation head
        # high_res_output = self.segmentation_head_high_res(decoder_output)
        # high_res_output = self.segmentation_head_high_res_dropout(high_res_output)


        # # Global average pooling and fully connected layer
        # global_avg = self.global_avg_pool(decoder_output)  # shape: [batch_size, 192, 1, 1]
        # global_avg = global_avg.view(global_avg.size(0), -1)  # Flatten to shape: [batch_size, 192]
        # global_avg = self.fc_global(global_avg)  # Pass through fc_global: shape [batch_size, 1]
        # global_avg = global_avg.unsqueeze(2).unsqueeze(3)  # Reshape to shape: [batch_size, 1, 1, 1]

        # # Expand global_avg to match the spatial dimensions of high_res_output
        # global_avg = global_avg.expand(-1, -1, high_res_output.size(2), high_res_output.size(3))

        # # Merge high resolution with global features
        # merged_features = torch.cat([high_res_output, global_avg], dim=1)
        # merged_features = self.fc_merge_dropout(merged_features)  # Apply dropout before merge

        # output = self.fc_merge(merged_features)

        return output

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x

class MyUnetPlusPlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], **kwargs
        )
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, inputFeatures):
        dense_xs = []
        for i in range(len(inputFeatures)):

            features = inputFeatures[i][1:]  # remove first skip with same spatial resolution
            features = features[::-1]  # reverse channels to start from head of encoder
            # start building dense connections
            dense_x = {}
            for layer_idx in range(len(self.in_channels) - 1):
                for depth_idx in range(self.depth - layer_idx):
                    if layer_idx == 0:
                        output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
                        dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                    else:
                        dense_l_i = depth_idx + layer_idx
                        cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
                        cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                        dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
                            dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features
                        )
            dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth-1}"])
            
            dense_xs.append(dense_x[f"x_{0}_{self.depth}"])
        return dense_xs
    
class MyUnetPlusPlus(torch.nn.Module):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        numberofEncoders: int = 1,
        device: str = "cpu",
        activation: Optional[Union[str, callable]] = None,
        decoderType : str = "unetplusplus",# unetplusplus or manet 
    ):
        super().__init__()

        if encoder_name.startswith("mit_b"):
            raise ValueError("UnetPlusPlus is not support encoder_name={}".format(encoder_name))

        self.encoders = []
        for i in range(numberofEncoders):
            self.encoders.append(get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=None,
            ).to(device))
        
        if decoderType == "unetplusplus":
            self.decoder = MyUnetPlusPlusDecoder(
                encoder_channels=self.encoders[0].out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                center=True if encoder_name.startswith("vgg") else False,
                attention_type=decoder_attention_type,
            )
        else:
            self.decoder = MyMAnetDecoder(
                encoder_channels=self.encoders[0].out_channels,
                decoder_channels=decoder_channels
            )

        self.segmentation_head = SegmentationHead(
            in_channels=numberofEncoders*decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.name = f"Myunetplusplus-{encoder_name} - {decoderType}"
        print(self.name)
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, x):
        x_split = torch.split(x, 1, dim=1)   

        features = []
        for i, encoder in enumerate(self.encoders):
            features.append(encoder(x_split[i].float()))

        features_concat = features

        decoder_output = self.decoder(features_concat)
        decoder_output = torch.cat(decoder_output, dim=1)
        masks = self.segmentation_head(decoder_output)

        return masks

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x
