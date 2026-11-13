# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math
from pytorch_wavelets import DWTForward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "CBAM",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "RepConv",
    "SpatialAttention",
    "DWTLSK",
    "SFDDown",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]: i[0] + 1, i[1]: i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, W, H) and output shape is (B, 4C, W/2, H/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SimpleBranch(nn.Module):
    """
    有序分支：简单一点
    先用 1 个普通 3x3 卷积
    """

    def __init__(self, c=3):
        super().__init__()
        self.block = ConvBNAct(c, c, k=3, s=1)

    def forward(self, x):
        return self.block(x)


class ComplexBranch(nn.Module):
    """
    无序分支：复杂一点
    先用 2 个普通 3x3 卷积
    """

    def __init__(self, c=3):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(c, c, k=3, s=1),
            ConvBNAct(c, c, k=3, s=1),
        )

    def forward(self, x):
        return self.block(x)


class DWTLSK(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1=3, c2=32, k=3, s=2, p=None, g=1, d=1, act=True, wave='haar'):
        super(DWTLSK, self).__init__()

        # 这版默认输入是RGB图像

        self.c1 = c1
        self.c2 = c2

        # 单层DWT
        self.dwt = DWTForward(J=1, mode='zero', wave=wave)

        # DWT拼接后通道数 = 4 * c1 = 12
        dwt_channels = 4 * c1
        self.lsk_gate = LSKGate(dwt_channels)
        # 4个有序分支（每组3通道）
        self.simple_branches = nn.ModuleList([SimpleBranch(c1) for _ in range(4)])

        # 4个无序分支（每组3通道）
        self.complex_branches = nn.ModuleList([ComplexBranch(c1) for _ in range(4)])

        # 8个可学习变量：4组 × 2个分支
        # alpha[i, 0] -> 第i组有序分支权重
        # alpha[i, 1] -> 第i组无序分支权重
        self.alpha = nn.Parameter(torch.ones(4, 2))

        # 拼接后再映射到输出通道 c2
        self.fuse = nn.Sequential(
            nn.Conv2d(dwt_channels, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # shortcut 分支：原图下采样后映射到 c2
        self.shortcut = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        yl, yh = self.dwt(x)

        lh = yh[0][:, :, 0, :, :]
        hl = yh[0][:, :, 1, :, :]
        hh = yh[0][:, :, 2, :, :]

        out = torch.cat([yl, lh, hl, hh], dim=1)  # [B, 12, H/2, W/2]

        # 用 LSK 生成 fore_weight
        fore_weight = self.lsk_gate(out)

        fore_img = out * fore_weight
        dif_img = out - fore_img

        fore_parts = fore_img.chunk(4, dim=1)
        dif_parts = dif_img.chunk(4, dim=1)

        alpha = torch.softmax(self.alpha, dim=1)

        fused_parts = []
        for i in range(4):
            fore_feat = self.simple_branches[i](fore_parts[i])
            dif_feat = self.complex_branches[i](dif_parts[i])

            fused_i = alpha[i, 0] * fore_feat + alpha[i, 1] * dif_feat
            fused_parts.append(fused_i)

        y = torch.cat(fused_parts, dim=1)
        y = self.fuse(y)
        shortcut = self.shortcut(x)

        return y + shortcut


class LSKGate(nn.Module):
    def __init__(self, channels):
        super().__init__()

        mid_channels = max(channels // 2, 4)

        # 新增：3x3 掩码卷积（建议用 depthwise，开销小）
        self.mask3 = MaskConv2d(
            channels, channels,
            kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        self.dw3 = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        # 原来的两条大核分支
        self.dw5 = nn.Conv2d(
            channels, channels,
            kernel_size=5, padding=2,
            groups=channels, bias=False
        )
        self.dw7_d3 = nn.Conv2d(
            channels, channels,
            kernel_size=7, padding=9, dilation=3,
            groups=channels, bias=False
        )

        # 三个分支分别压缩通道
        self.pw0 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.pw1 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.pw2 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)

        # 原来输出2个选择图，现在输出3个
        self.selector = nn.Conv2d(2, 3, kernel_size=7, padding=3, bias=True)

        # 投影回原通道
        self.proj = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        # 三层递进式特征
        a0 = self.dw3(x)  # 局部差异/边缘
        a1 = self.dw5(a0)  # 中等感受野
        a2 = self.dw7_d3(a1)  # 更大感受野

        # 各自压缩
        a0 = self.pw0(a0)
        a1 = self.pw1(a1)
        a2 = self.pw2(a2)

        # 聚合统计信息
        a = torch.cat([a0, a1, a2], dim=1)  # [B, 3*mid, H, W]
        avg = torch.mean(a, dim=1, keepdim=True)
        mx, _ = torch.max(a, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)  # [B, 2, H, W]

        # 生成3个分支权重
        s = self.selector(s)  # [B, 3, H, W]
        s = torch.softmax(s, dim=1)  # 更适合多分支竞争选择

        # 三分支自适应融合
        out = (
                a0 * s[:, 0:1, :, :] +
                a1 * s[:, 1:2, :, :] +
                a2 * s[:, 2:3, :, :]
        )

        # 输出 gate
        out = torch.sigmoid(self.proj(out))
        return out


class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )

        mask = torch.ones_like(self.weight)
        kh, kw = self.weight.shape[-2:]
        mask[:, :, kh // 2, kw // 2] = 0.0  # 中心置0
        self.register_buffer("mask", mask)

    def forward(self, x):
        weight = self.weight * self.mask
        return F.conv2d(
            x, weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )

    class AConvLKS(nn.Module):
        """
        AConv + lightweight LKS
        结构：
            AvgPool(2,1) -> Conv(3x3,s=2) -> DW Dilated Conv(3x3,d=d)
        说明：
            1. 先平滑再下采样，减少信息突变
            2. 下采样后接一个轻量DW膨胀卷积，扩大感受野
            3. 参数增加很小：仅增加 9*c2 个depthwise参数
        """

        def __init__(self, c1, c2, d=2):
            super().__init__()
            self.cv1 = Conv(c1, c2, 3, 2, 1)
            self.dw = Conv(c2, c2, 3, 1, None, g=c2, d=d)  # depthwise dilated conv

        def forward(self, x):
            x = F.avg_pool2d(x, 2, 1, 0, False, True)
            x = self.cv1(x)
            x = self.dw(x)
            return x

    class ADownLKS(nn.Module):
        """
        ADown + lightweight LKS
        结构：
            AvgPool(2,1)
              -> split
                 branch1: Conv(3x3,s=2) -> DW Dilated Conv(3x3,d=d)
                 branch2: MaxPool(3,2,1) -> Conv(1x1)
              -> Concat

        说明：
            1. 保留 ADown 的双分支信息保留能力
            2. 只在卷积分支上加 LKS，控制参数和计算量
            3. 比在 concat 后整块做 LKS 更轻
        """

        def __init__(self, c1, c2, d=2):
            super().__init__()
            assert c1 % 2 == 0, f"ADownLKS expects even input channels, but got c1={c1}"
            self.c = c2 // 2

            self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
            self.dw = Conv(self.c, self.c, 3, 1, None, g=self.c, d=d)  # depthwise dilated conv
            self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

        def forward(self, x):
            x = F.avg_pool2d(x, 2, 1, 0, False, True)
            x1, x2 = x.chunk(2, 1)

            y1 = self.cv1(x1)
            y1 = self.dw(y1)

            y2 = F.max_pool2d(x2, 3, 2, 1)
            y2 = self.cv2(y2)

            return torch.cat((y1, y2), 1)


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    b, c, h, w = x.shape
    assert c % groups == 0
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class SFDDown(nn.Module):
    """
    Small-object frequency-preserving downsampling.

    This module avoids using a single stride-2 convolution for UAV/small-object
    features. It fuses anti-aliased low-frequency features, lossless
    space-to-depth samples, and Haar high-frequency details with a lightweight
    content gate.
    """

    def __init__(self, c1, c2, k=3, s=2, e=0.5, use_gate=True):
        super().__init__()
        assert s == 2, "SFDDown is designed for stride-2 downsampling"

        c_mid = max(int(c2 * e), 24)
        self.branch_c = max(c_mid // 3, 8)
        self.use_gate = use_gate

        self.low_proj = Conv(c1, self.branch_c, 1, 1)
        self.low_dw = Conv(self.branch_c, self.branch_c, k, 1, g=self.branch_c)

        self.spd_proj = Conv(4 * c1, self.branch_c, 1, 1)
        self.spd_dw = Conv(self.branch_c, self.branch_c, k, 1, g=self.branch_c)

        self.high_proj = Conv(3 * c1, self.branch_c, 1, 1)
        self.high_dw = Conv(self.branch_c, self.branch_c, k, 1, g=self.branch_c)

        if use_gate:
            hidden = max(self.branch_c, 16)
            self.gate = nn.Sequential(
                nn.Conv2d(6 * self.branch_c, hidden, 1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, 3 * self.branch_c, 1, bias=True),
            )

        self.fuse = Conv(3 * self.branch_c, c2, 1, 1)
        self.short = Conv(c1, c2, 1, 1, act=False)

        blur = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ) / 16.0
        self.register_buffer("blur_kernel", blur.view(1, 1, 3, 3), persistent=False)

    @staticmethod
    def _pad_even(x):
        _, _, h, w = x.shape
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x

    def _blur_down(self, x):
        x = self._pad_even(x)
        c = x.shape[1]
        weight = self.blur_kernel.to(device=x.device, dtype=x.dtype).expand(c, 1, 3, 3)
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(x, weight, stride=2, groups=c)

    def _haar_high(self, x):
        x = self._pad_even(x)
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return torch.cat((lh, hl, hh), dim=1)

    def forward(self, x):
        low = self.low_dw(self.low_proj(self._blur_down(x)))
        spd = self.spd_dw(self.spd_proj(F.pixel_unshuffle(self._pad_even(x), 2)))
        high = self.high_dw(self.high_proj(self._haar_high(x)))

        if self.use_gate:
            m = torch.cat((low, spd, high), dim=1)
            stat = torch.cat(
                (F.adaptive_avg_pool2d(m, 1), F.adaptive_max_pool2d(m, 1)),
                dim=1,
            )
            gate = self.gate(stat).view(x.shape[0], 3, self.branch_c, 1, 1)
            gate = torch.softmax(gate, dim=1)
            out = torch.cat(
                (low * gate[:, 0], spd * gate[:, 1], high * gate[:, 2]),
                dim=1,
            )
        else:
            out = torch.cat((low, spd, high), dim=1)

        return self.fuse(out) + self.short(self._blur_down(x))
