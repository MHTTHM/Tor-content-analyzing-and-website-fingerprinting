"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}

# Resnet



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def _log_api_usage_once(module: str, name: str) -> None:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return
    torch._C._log_api_usage_once(f"torchvision.{module}.{name}")

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        _log_api_usage_once("models", self.__class__.__name__)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x

        

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)

    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)



"""
Refering to https://github.com/JunwenBai/c-gmvae/blob/master/model.py
"""
class MLP(nn.Module):
    def __init__(self, in_dim, dropout=0.2):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

    
    def forward(self, x):
        # x形状[64, 200]
        # print("MLP input shape: ", x.shape)
        # print("MLP output shape: ", self.mlp(x).shape)
        return self.mlp(x)


import copy

class Transformer(nn.Module):

    def __init__(self, in_dim, nhead=8, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256*4, dropout=0.1):
        super().__init__()

        self.encoder = TransformerEncoder(TransformerEncoderLayer(in_dim, nhead, dim_feedforward, dropout),
                                          num_encoder_layers)
        self.decoder = TransformerDecoder(TransformerDecoderLayer(in_dim, nhead, dim_feedforward, dropout),
                                          num_decoder_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):
        bs = src.shape[0]
        #print("Transformer src shape: ", src.shape)
        #print("Transformer pos shape: ", pos_embed.shape)
        memory = self.encoder(src, pos_embed)
        query_embed = query_embed.repeat(bs, 1, 1)
        tgt = torch.zeros_like(query_embed)
        output = self.decoder(tgt, memory, pos_embed, query_embed)
        return output


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, pos):
        #print("encoder src shape: ", src.shape)
        #print("encoder pos shape: ", pos.shape)
        output = src
        for layer in self.layers:
            output = layer(output, pos)
        
        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos, query_pos):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos, query_pos)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, src, pos):
        #print("encoder layer src shape: ", src.shape)
        #print("encoder layer pos shape: ", pos.shape)
        src2 = self.self_attn(query=src + pos, key=src + pos, value=src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, tgt, memory, pos, query_pos):
        tgt2 = self.self_attn(query=tgt + query_pos, key=tgt + query_pos, value=tgt)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt + query_pos, key=memory + pos, value=memory)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class DFNet(nn.Module):
    def __init__(self, dropout):
        super(DFNet, self).__init__()

        # Block1
        filter_num = [0, 32, 64, 128, 256]
        kernel_size = [0, 8, 8, 8, 8]
        conv_stride_size = [0, 1, 1, 1, 1]
        pool_stride_size = [0, 4, 4, 4, 4]
        pool_size = [0, 8, 8, 8, 8]

        self.block1_conv1 = nn.Conv1d(in_channels=1, out_channels=filter_num[1],
                                      kernel_size=kernel_size[1],
                                      stride=conv_stride_size[1], padding=kernel_size[1] // 2)
        self.block1_bn1 = nn.BatchNorm1d(num_features=filter_num[1])
        self.block1_elu1 = nn.ELU(alpha=1.0)
        self.block1_conv2 = nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[1], kernel_size=kernel_size[1],
                                      stride=conv_stride_size[1], padding=kernel_size[1] // 2)
        self.block1_bn2 = nn.BatchNorm1d(num_features=filter_num[1])
        self.block1_elu2 = nn.ELU(alpha=1.0)
        self.block1_pool = nn.MaxPool1d(kernel_size=pool_size[1], stride=pool_stride_size[1], padding=pool_size[1] // 2)
        self.block1_dropout = nn.Dropout(p=dropout)

        self.block2_conv1 = nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[2], kernel_size=kernel_size[2],
                                      stride=conv_stride_size[2], padding=kernel_size[2] // 2)
        self.block2_bn1 = nn.BatchNorm1d(num_features=filter_num[2])
        self.block2_relu1 = nn.ReLU()
        self.block2_conv2 = nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[2], kernel_size=kernel_size[2],
                                      stride=conv_stride_size[2], padding=kernel_size[2] // 2)
        self.block2_bn2 = nn.BatchNorm1d(num_features=filter_num[2])
        self.block2_relu2 = nn.ReLU()
        self.block2_pool = nn.MaxPool1d(kernel_size=pool_size[2], stride=pool_stride_size[2], padding=pool_size[2] // 2)
        self.block2_dropout = nn.Dropout(p=dropout)

        self.block3_conv1 = nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[3], kernel_size=kernel_size[3],
                                      stride=conv_stride_size[3], padding=kernel_size[3] // 2)
        self.block3_bn1 = nn.BatchNorm1d(num_features=filter_num[3])
        self.block3_relu1 = nn.ReLU()
        self.block3_conv2 = nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[3], kernel_size=kernel_size[3],
                                      stride=conv_stride_size[3], padding=kernel_size[3] // 2)
        self.block3_bn2 = nn.BatchNorm1d(num_features=filter_num[3])
        self.block3_relu2 = nn.ReLU()
        self.block3_pool = nn.MaxPool1d(kernel_size=pool_size[3], stride=pool_stride_size[3], padding=pool_size[3] // 2)
        self.block3_dropout = nn.Dropout(p=dropout)

        self.block4_conv1 = nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[4], kernel_size=kernel_size[4],
                                      stride=conv_stride_size[4], padding=kernel_size[4] // 2)
        self.block4_bn1 = nn.BatchNorm1d(num_features=filter_num[4])
        self.block4_relu1 = nn.ReLU()
        self.block4_conv2 = nn.Conv1d(in_channels=filter_num[4], out_channels=filter_num[4], kernel_size=kernel_size[4],
                                      stride=conv_stride_size[4], padding=kernel_size[4] // 2)
        self.block4_bn2 = nn.BatchNorm1d(num_features=filter_num[4])
        self.block4_relu2 = nn.ReLU()
        self.block4_pool = nn.MaxPool1d(kernel_size=pool_size[4], stride=pool_stride_size[4], padding=pool_size[4] // 2)
        self.block4_dropout = nn.Dropout(p=dropout)

    def forward(self, input):

        if len(input.shape) == 2:
            x = input.unsqueeze(1)
        else:
            x = input

        # Block 1
        x = self.block1_conv1(x)
        x = self.block1_bn1(x)
        x = self.block1_elu1(x)
        x = self.block1_conv2(x)
        x = self.block1_bn2(x)
        x = self.block1_elu2(x)
        x = self.block1_pool(x)
        x = self.block1_dropout(x)

        # Block 2
        x = self.block2_conv1(x)
        x = self.block2_bn1(x)
        x = self.block2_relu1(x)
        x = self.block2_conv2(x)
        x = self.block2_bn2(x)
        x = self.block2_relu2(x)
        x = self.block2_pool(x)
        x = self.block2_dropout(x)

        # Block 3
        x = self.block3_conv1(x)
        x = self.block3_bn1(x)
        x = self.block3_relu1(x)
        x = self.block3_conv2(x)
        x = self.block3_bn2(x)
        x = self.block3_relu2(x)
        x = self.block3_pool(x)
        x = self.block3_dropout(x)

        # Block 4
        x = self.block4_conv1(x)
        x = self.block4_bn1(x)
        x = self.block4_relu1(x)
        x = self.block4_conv2(x)
        x = self.block4_bn2(x)
        x = self.block4_relu2(x)
        x = self.block4_pool(x)
        x = self.block4_dropout(x)
        return x.transpose(1, 2)

class TMWF_DFNet(nn.Module):

    def __init__(self, in_dim,ebed_dim=256, nhead=8, dim_feedforward=256*4, num_encoder_layers=2, num_decoder_layers=2, max_len=21, num_queries=2,
                 cls=51, dropout=0.1):
        super(TMWF_DFNet, self).__init__()
        print('TMWF_DFNet')
        self.cnn_layer = DFNet(dropout)
        self.proj = nn.Sequential(
            nn.Linear(ebed_dim, ebed_dim),
            nn.LayerNorm(ebed_dim)
        )
        self.trm = Transformer(ebed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.pos_embed = nn.Embedding(max_len, ebed_dim).weight
        self.query_embed = nn.Embedding(num_queries, ebed_dim).weight
        self.fc = nn.Linear(ebed_dim, cls)

    def forward(self, input):
        #print("input shape: ", input.shape)
        x = self.cnn_layer(input)
        #print("x shape: ", x.shape)
        feat = self.proj(x)
        #print("feat shape: ", feat.shape)
        #print("query shape: ", self.query_embed.unsqueeze(0).shape)
        #print("pos shape: ", self.pos_embed.unsqueeze(0).shape)
        o = self.trm(feat, self.query_embed.unsqueeze(0), self.pos_embed.unsqueeze(0))
        o = o.squeeze(1)
        #print("o shape: ", o.shape)
        #logits = self.fc(o)
        #print("logits shape: ", logits.shape)
        return o


#################################################################
import numpy as np
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath, Mlp

class TopM_MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_mhsa_layers, dim_feedforward, dropout, top_m):
        super().__init__()

        self.nets = nn.ModuleList([MHSA_Block(embed_dim, num_heads, dim_feedforward, dropout, top_m) for _ in range(num_mhsa_layers)])

    def forward(self, x, pos_embed):
        output = x + pos_embed
        for layer in self.nets:
            output = layer(output)
        return output

class TopMAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout, top_m):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.top_m = top_m

        self.qkv = nn.Linear(dim , dim*3)
        self.attn_drop = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout),
        )
        self.proj_drop = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.zeros(B, self.num_heads, N, N, device=q.device, requires_grad=False)
        index = torch.topk(attn, k=self.top_m, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask>0, attn, torch.full_like(attn, float('-inf')))

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj_drop(x)
        return x

class MHSA_Block(nn.Module):

    def __init__(self, embed_dim, nhead, dim_feedforward, dropout, top_m):
        super().__init__()
        drop_path_rate = 0.1
        self.attn = TopMAttention(embed_dim, nhead, dropout, top_m)
        self.drop_path = DropPath(drop_path_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=dim_feedforward, act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(ConvBlock1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,kernel_size=kernel_size,dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        self.last_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)


class LocalProfiling(nn.Module):
    """ Local Profiling module in ARES """
    def __init__(self):
        super(LocalProfiling, self).__init__()
        
        self.net = nn.Sequential(
            ConvBlock1d(in_channels=1, out_channels=32, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=32, out_channels=64, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=64, out_channels=128, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=128, out_channels=256, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        x = self.net(x)
        return x

class Trans_WF(nn.Module):
    """ Trans_WF in ARES """
    def __init__(self, in_dim, num_classes=51, dropout=0.2):
        super(Trans_WF, self).__init__()

        embed_dim = 256
        num_heads = 8
        dim_feedforward = 256 * 4
        num_mhsa_layers = 2
        dropout = 0.1
        max_len = 32
        top_m = 20
        
        self.dividing = nn.Sequential(
            Rearrange('b c (n p) -> (b n) c p', n=4),
        )
        self.combination = nn.Sequential(
            Rearrange('(b n) c p -> b c (n p)', n=4),
        )
        self.profiling = LocalProfiling()
        self.pos_embed = nn.Embedding(max_len, embed_dim).weight

        self.topm_mhsa = TopM_MHSA(embed_dim, num_heads, num_mhsa_layers, dim_feedforward, dropout, top_m)
        self.mlp = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        sliding_size = np.random.randint(0, 1 + 2500)
        x = x.unsqueeze(1)
        x = torch.roll(x, shifts=sliding_size, dims=-1)
        x = self.dividing(x)
        x = self.profiling(x)
        x = self.combination(x)
        x = x.permute(0, 2, 1)
        x = self.topm_mhsa(x, self.pos_embed.unsqueeze(0))
        x = x.mean(dim=1)
        #x = self.mlp(x)
        return x


# CNN with 2 dim input
class CNNClassifier(nn.Module):
    def __init__(self, in_dim, in_channels=2,  dropout=0.2):
        super(CNNClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),  # 1D卷积
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 最大池化
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # 计算卷积和池化后的序列长度
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * (in_dim // 4), 256),  # 假设经过两次池化，序列长度变为 seq_len / 4
            nn.ReLU(),
            nn.Dropout(p=dropout),
            #nn.Linear(256, num_classes)  # 输出类别数
        )

    def forward(self, x):
        # x shape: (batch_size, in_channels, seq_len)
        x = self.conv_layers(x)  # 通过卷积层
        x = self.flatten(x)  # 展平为 (batch_size, 128 * (seq_len // 4))
        x = self.fc(x)  # 通过全连接层
        return x  # 输出形状: (batch_size, num_classes)