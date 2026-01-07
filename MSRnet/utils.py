import os
import torch.nn as nn
from torch.nn import functional as F
import torch
import datetime
import logging
import shutil
import numpy as np



class CE_Label_Smooth_Loss(nn.Module):
    def __init__(self, classes=4, epsilon=0.2):
        super(CE_Label_Smooth_Loss, self).__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss



def set_logging_config(logdir):
    def beijing(sec, what):
        beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
        return beijing_time.timetuple()


    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logging.Formatter.converter = beijing

    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, ("log.txt"))),
                                  logging.StreamHandler(os.sys.stdout)])

#SE_Block（Squeeze-and-Excitation Block）是一种 注意力机制，
# 用于 通道权重自适应调整。它会 重新计算各通道的重要性，并增强关键通道的特征表示。
class SE_Block(nn.Module):
    def __init__(self, inchannel):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel),
            nn.ReLU(),
            nn.Linear(inchannel, inchannel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def normalize(w):
    if len(w.shape) == 4:
        d = torch.sum(torch.abs(w), dim=3)
    elif len(w.shape) == 3:
        d = torch.sum(torch.abs(w), dim=2)

    d_re = 1 / torch.sqrt(d + 1e-5)
    d_re[d_re == float('inf')] = 0
    d_matrix = torch.diag_embed(d_re)

    return torch.matmul(torch.matmul(d_matrix, w), d_matrix)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes=62, ratio=16):
        super(ChannelAttention, self).__init__()
        reduced_planes = max(1, in_planes // ratio)

        self.fc1 = nn.Conv1d(in_planes, reduced_planes, 1, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)  # Xavier 初始化
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(reduced_planes, in_planes, 1, bias=False)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(torch.mean(x, dim=2, keepdim=True))))  # 平均池化
        max_out = self.fc2(self.relu1(self.fc1(torch.max(x, dim=2, keepdim=True)[0])))  # 最大池化
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=1):  # 使用较小卷积核，减少复杂度
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight)  # Xavier 初始化
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels=62, ratio=5):
        super(CBAM, self).__init__()
        self.ChannelAttention = ChannelAttention(in_channels, ratio)
        self.SpatialAttention = SpatialAttention(kernel_size=1)  # 使用较小卷积核

    def forward(self, x):
        # 改为并行处理通道和空间注意力
        channel_attention_out = self.ChannelAttention(x) * x
        spatial_attention_out = self.SpatialAttention(x) * x
        x = channel_attention_out + spatial_attention_out  # 直接融合
        return x

# 加入Dropout和正则化
class CBAMWithRegularization(nn.Module):
    def __init__(self, in_channels=62, ratio=5, dropout_rate=0.1):
        super(CBAMWithRegularization, self).__init__()
        self.cbam = CBAM(in_channels, ratio)
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout

    def forward(self, x):
        x = self.cbam(x)
        x = self.dropout(x)  # 应用Dropout
        return x

def feature_trans(subgraph_num, feature):
    return feature
    # if subgraph_num == 7:
    #     return feature_trans_7(feature)

# def feature_trans_7(feature):
#     reassigned_feature = torch.cat((
#         feature[:, 0:5],
#
#         feature[:, 5:8], feature[:, 14:17], feature[:, 23:26],
#
#         feature[:, 23:26], feature[:, 32:35], feature[:, 41:44],
#
#         feature[:, 7:12], feature[:, 16:21], feature[:, 25:30],
#         feature[:, 34:39], feature[:, 43:48],
#
#         feature[:, 11:14], feature[:, 20:23], feature[:, 29:32],
#
#         feature[:, 29:32], feature[:, 38:41], feature[:, 47:50],
#
#         feature[:, 50:62]), dim=1)
#
#     return reassigned_feature

    return reassigned_feature
# utils.py 中新增
import torch.nn.functional as F
import torch.nn as nn
import torch

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device, lambda_c=1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c
        self.device = device

        # 初始化类中心
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        batch_size = features.size(0)
        expanded_centers = self.centers.index_select(dim=0, index=labels)
        loss = F.mse_loss(features, expanded_centers)
        return self.lambda_c * loss