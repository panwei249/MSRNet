import torch.nn as nn
from layer_MIND import State_encoder, Self_attention, Region
import torch
import numpy as np
from einops import rearrange

class ResidualFusion(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualFusion, self).__init__()
        self.attention1 = nn.MultiheadAttention(embed_dim=in_dim, num_heads=6, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=in_dim, num_heads=6, batch_first=True)
        self.fc = nn.Linear(in_dim * 2, out_dim)

    def forward(self, f1, f2):
        # 通过自注意力机制处理两个特征
        attn_f1, _ = self.attention1(f1, f2, f2)
        attn_f2, _ = self.attention2(f2, f1, f1)

        # 残差连接：将注意力后的特征与原始特征相加
        fused_f1 = attn_f1 + f1
        fused_f2 = attn_f2 + f2

        # 合并残差后的特征
        combined = torch.cat((fused_f1, fused_f2), dim=-1)

        # 通过全连接层生成最终输出
        out = self.fc(combined)
        return out
class Mind(nn.Module):
    def __init__(self, args, cluster_labels=None):
        super(Mind, self).__init__()

        self.args = args
        self.cluster_labels = cluster_labels
        self.nclass = args.n_class
        self.dropout = args.dropout

        self.selu = nn.SELU()
        self.at = nn.SELU()
        self.embed = nn.Linear(50, 60)

        self.encoder = State_encoder(args.in_feature, 50)

        cluster_labels_torch = torch.tensor(self.cluster_labels, dtype=torch.long)
        self.region = Region(
            cluster_labels=cluster_labels_torch,
            trainable_vector=62,
            emb_size=50
        )

        self.sa = Self_attention(60, 85)

        self.res_fusion = ResidualFusion(in_dim=60, out_dim=60)

        # 最终 MLP
        self.mlp1 = nn.Linear(4140, 2048)
        self.mlp2 = nn.Linear(2048, 512)
        self.mlp3 = nn.Linear(512, self.nclass)

        self.BN1 = nn.BatchNorm1d(5)
        self.BN2 = nn.BatchNorm1d(7)
        self.BN3 = nn.BatchNorm1d(62)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(self.dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 如果是 SEED5，需要 reshape
        if self.args.dataset == 'SEED5':
            x = rearrange(x, 'b (h c) -> b h c', h=62)

        # 1) Local branch
        vq_loss, local_x1, usage_local, codebook_local = self.encoder(x)
        res_local = self.at(self.embed(local_x1))

        # 2) Region branch
        region_loss, region_x, usage_reg, codebook_region = self.region(x)
        region_x = self.at(region_x)

        region_loss1, region_x1, usage_reg1, codebook_region1 = self.region(x)
        region_x1 = self.at(region_x1)

        usage = [usage_local] + usage_reg
        codebook = [codebook_local] + codebook_region

        expanded_region_x = torch.zeros((region_x.size(0), 62, region_x.size(2)), device=region_x.device)

        # 遍历每个区域，将 region_x 的信息复制到相应的通道位置
        for i in range(62):
            region_idx = self.cluster_labels[i]  # 获取该通道对应的区域
            expanded_region_x[:, i, :] = region_x[:, region_idx, :]
        fused_output = self.res_fusion(res_local, expanded_region_x)
        # print(fused_output.shape)
        fused_output = torch.concat((fused_output, region_x), dim=1)
        # fused_output = torch.concat((expanded_region_x, fused_output), dim=2)

        # 4) flatten + MLP
        x = fused_output.view(fused_output.size(0), -1)
        # x = torch.concat((res_local, region_x), dim=1)
        # x_ = self.at(self.sa(x))
        # x = torch.concat((x, x_), dim=2)
        #
        #
        #

        # x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.at(self.bn1(self.mlp1(x)))
        x = self.dropout(x)
        x = self.at(self.bn2(self.mlp2(x)))
        x = self.dropout(x)
        x = self.mlp3(x)

        return x, vq_loss, region_loss, usage, codebook,fused_output.view(fused_output.size(0), -1)

