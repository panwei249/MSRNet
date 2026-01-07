# -*- coding: utf-8 -*-
"""Multi-scale 1D convolution modules for EEG.

This file is split out from the original `pasted.txt`:
- CAB: Channel Attention Block
- SAB: Spatial/Temporal Attention Block (self-attention over time)
- MSCB: Multi-Scale Convolution Block
- EnhancedTemporalLearner: stacks MSCB blocks and outputs a pooled feature vector
"""

import torch
import torch.nn as nn

class CAB(nn.Module):
    """通道注意力模块（适用于1D卷积）"""
    def __init__(self, channels, reduction=16):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * out

class SAB(nn.Module):
    """空间（时间）注意力模块（适用于1D卷积）"""
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(x_cat))
        return x * scale

class MSCB(nn.Module):
    """多尺度卷积模块（适用于1D卷积）"""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7]):
        super(MSCB, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, k, padding=k//2, groups=in_channels, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ) for k in kernel_sizes
        ])
        self.pointwise = nn.Sequential(
            nn.Conv1d(len(kernel_sizes)*out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = torch.cat([conv(x) for conv in self.convs], dim=1)
        out = self.pointwise(out)
        return out


class EnhancedTemporalLearner(nn.Module):
    """
    融合多尺度与注意力机制的1D卷积特征提取模块（输出维度与原版保持一致）
    输入: (B, T, 1)
    输出: (B, out_channels * num_kernels)
    """

    def __init__(self, kernel_sizes=[3, 5, 7], out_channels=50):
        super(EnhancedTemporalLearner, self).__init__()
        self.initial_conv = nn.Conv1d(1, out_channels, 1)
        self.cab = CAB(out_channels)
        self.sab = SAB()

        # 注意这里修改 MSCB 输出通道数
        self.mscb = MSCB(out_channels, out_channels, kernel_sizes)

        # BN和ReLU层修改为对应的输出维度（out_channels * num_kernels）
        self.bn = nn.BatchNorm1d(out_channels * len(kernel_sizes))
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (B, T, 1)
        x = x.transpose(1, 2)  # (B, 1, T)
        x = self.initial_conv(x)  # (B, out_channels, T)
        x = self.cab(x)  # 通道注意力 (B, out_channels, T)
        x = self.sab(x)  # 空间注意力 (B, out_channels, T)

        # 改为直接拼接多尺度特征
        multi_scale_feats = [conv(x) for conv in self.mscb.convs]  # 每个尺度 (B,out_channels,T)
        x_cat = torch.cat(multi_scale_feats, dim=1)  # 拼接 (B,150,T)

        x = self.bn(x_cat)
        x = self.relu(x)
        out = x.mean(dim=2)  # 平均池化 (B, 150)
        return out
