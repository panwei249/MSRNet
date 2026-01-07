import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from utils import SE_Block, normalize, CBAM, feature_trans,CBAMWithRegularization
# -------- 新的 DynamicCodebook 替代原 Codebook
# (见下方 DynamicCodebook 定义)
# -----------------------------------------
from einops import rearrange

# 注意：以下 import 如果你单独拆文件，需要改成相对 import
# from . import DynamicCodebook
from Codebook import DynamicCodebook
class get_adj(nn.Module):
    def __init__(self, in_features, num_chan, num_embedding, embedding_dim):
        super(get_adj, self).__init__()
        self.in_features = in_features
        self.num_chan = num_chan
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim

        self.p = nn.Parameter(torch.empty(self.num_chan, self.num_chan))
        self.bias = nn.Parameter(torch.empty(self.num_chan, self.in_features))

        self.q = nn.Parameter(torch.empty(self.in_features, self.in_features))
        self.theta = nn.Parameter(torch.empty(self.in_features, self.num_chan * self.in_features))

        self.at = nn.ELU()
        self.senet = SE_Block(self.in_features)

        # ============= 改成动态码本 =============
        self.cb = DynamicCodebook(
            in_chan=self.in_features,
            num_embeddings=self.num_embedding,
            input_chan=self.num_chan,
            embedding_dim=self.embedding_dim,
            commitment_cost=0.25,
            usage_threshold=1e-3  # 可自定义
        )
        # =====================================

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.p)
        nn.init.zeros_(self.bias)
        nn.init.xavier_normal_(self.q)
        nn.init.xavier_normal_(self.theta)

    def forward(self, x):
        # x: [batch, channels, features]

        o = torch.einsum("ij, b j d->b i d", self.p, x) + self.bias
        g = self.at(torch.matmul(torch.matmul(o, self.q), self.theta))
        adj = normalize(g.reshape(g.shape[0], g.shape[1], g.shape[1], -1).permute(0, 3, 1, 2))
        vq_loss, adj_recon, usage, codebook = self.cb(adj)

        adj_recon = self.at(self.senet(adj_recon) + adj_recon)
        adj_recon = normalize(self.at(torch.sum(adj_recon, dim=1)))
        return vq_loss, adj_recon, usage, codebook


class res_gcn(nn.Module):
    def __init__(self, in_features, num_chan, dp=0.01):
        super(res_gcn, self).__init__()
        self.in_features = in_features
        self.num_chan = num_chan

        self.weight1 = Parameter(torch.empty(self.in_features, self.in_features + 10))
        self.weight2 = Parameter(torch.empty(self.in_features + 15, self.in_features + 25))
        self.CBAM1 = CBAM(self.num_chan)
        self.CBAM2 = CBAM(self.num_chan)

        self.dropout = nn.Dropout(p=dp)
        self.dp = nn.Dropout(0.1)
        self.at = nn.SELU()
        self.init_weight()

    def init_weight(self):
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv1, stdv1)
        stdv2 = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv2, stdv2)

    def forward(self, x, adj):
        adj1 = self.dp(adj)
        x1_ = torch.einsum("b i j, j d -> b i d", x, self.weight1)
        x1 = self.at(torch.einsum("b i j, b j d->b i d", adj1, x1_))
        x1 = torch.concat((x, x1), 2)
        x1 = self.CBAM1(x1)
        x11 = self.dropout(x1)

        x2_ = torch.einsum("b i j, j d -> b i d", x11, self.weight2)
        x2 = self.at(torch.einsum("b i j, b j d->b i d", adj1, x2_))
        x2 = torch.concat((x1, x2), 2)
        x2 = self.CBAM2(x2)
        return x2


class res_gcn2(nn.Module):
    def __init__(self, in_features, k):
        super(res_gcn2, self).__init__()
        self.in_features = in_features
        self.weight1 = Parameter(torch.empty(self.in_features, self.in_features + 5))
        self.weight2 = Parameter(torch.empty(self.in_features + 5, self.in_features + 10))
        self.CBAM1 = CBAM(7)
        self.CBAM2 = CBAM(7)
        self.dropout = nn.Dropout(p=0.01)
        self.dp = nn.Dropout(0.1)
        self.at = nn.SELU()
        self.init_weight()

    def init_weight(self):
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv1, stdv1)
        stdv2 = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv2, stdv2)

    def forward(self, x, adj):
        adj1 = self.dp(adj)
        x1_ = torch.einsum("b i j, j d -> b i d", x, self.weight1)
        x1 = self.at(torch.einsum("b i j, b j d->b i d", adj1, x1_))
        x1 = self.CBAM1(x1) + x1
        x11 = self.dropout(x1)

        x2_ = torch.einsum("b i j, j d -> b i d", x11, self.weight2)
        x2 = self.at(torch.einsum("b i j, b j d->b i d", adj1, x2_))
        x2 = self.CBAM2(x2) + x2
        return x2


class State_encoder(Module):
    def __init__(self, in_features, out_features):
        super(State_encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.get_adj = get_adj(5, 62, 64, 1024)
        self.res_gcn = res_gcn(5, 62, 0.1)

    def forward(self, x):
        vq_loss, adj_recon, usage, codebook_local = self.get_adj(x)
        x = self.res_gcn(x, adj_recon)
        return vq_loss, x, usage, codebook_local


class Self_attention(nn.Module):
    def __init__(self, in_features, out_feature):
        super(Self_attention, self).__init__()
        self.in_features = in_features
        self.out_features = out_feature
        self.num_heads = 5
        self.selu = nn.SELU()

        self.get_qk = nn.Linear(self.in_features, self.in_features * 2)
        nn.init.xavier_uniform_(self.get_qk.weight)

        self.equ_weights = Parameter(torch.FloatTensor(self.num_heads))
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.bias = Parameter(torch.FloatTensor(self.out_features))
        self.reset_parameters()

    def forward(self, h):
        h_ = self.cal_att_matrix(h)
        output = torch.matmul(h_, self.weight) + self.bias
        return output

    def cal_att_matrix(self, feature):
        qk = rearrange(self.get_qk(feature), "b n (h d qk) -> (qk) b h n d", h=self.num_heads, qk=2)
        queries, keys = qk[0], qk[1]
        values = feature
        dim_scale = (queries.size(-1)) ** -0.5
        dots = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) * dim_scale
        at = torch.einsum("b g i j -> b i j", dots)
        adj_matrix = self.dropout_layer(at)
        at = F.softmax(adj_matrix / 0.3, dim=2)
        out_feature = torch.einsum('b i j, b j d -> b i d', at, values)
        return out_feature

    def dropout_layer(self, at):
        att_subview_, _ = at.sort(2, descending=True)
        att_threshold = att_subview_[:, :, att_subview_.size(2) // 6]
        att_threshold = rearrange(att_threshold, 'b i -> b i 1')
        att_threshold = att_threshold.repeat(1, 1, at.size()[2])
        at[at < att_threshold] = -1e-7
        return at

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.equ_weights.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)


# ============================
# Region 动态子图划分
# ============================
import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from einops import rearrange

class Region(nn.Module):
    def __init__(self, cluster_labels, trainable_vector, emb_size=50):
        super(Region, self).__init__()
        self.cluster_labels = torch.tensor(cluster_labels, dtype=torch.long)
        self.subgraph_num = int(self.cluster_labels.max().item() + 1)

        self.at = nn.ELU()
        self.emb_size = emb_size

        self.get_adj_list = nn.ModuleList()
        self.res_gcn_list = nn.ModuleList()

        self.subgraph_sizes = []
        for c_idx in range(self.subgraph_num):
            c_size = (self.cluster_labels == c_idx).sum().item()
            self.subgraph_sizes.append(c_size)

        # 按每个子图大小初始化
        from layer_MIND import get_adj, res_gcn, res_gcn2
        for size in self.subgraph_sizes:
            self.get_adj_list.append(
                get_adj(in_features=5, num_chan=size, num_embedding=32, embedding_dim=16)
            )
            self.res_gcn_list.append(
                res_gcn(in_features=5, num_chan=size)
            )

        self.get_adj_co = get_adj(50, self.subgraph_num, 128, 32)
        self.res_gcn_co = res_gcn2(50, 2)

        self.trainable_vec1 = Parameter(torch.FloatTensor(trainable_vector))
        self.weight1 = Parameter(torch.FloatTensor(self.emb_size, 80))
        self.softmax = nn.Softmax(dim=0)
        self.att_softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.trainable_vec1.size(0))
        self.trainable_vec1.data.uniform_(-stdv1, stdv1)
        self.weight1.data.uniform_(-stdv1, stdv1)

    def forward(self, x):
        sub_features = self.region_x(x)

        usage_list = []
        codebook_list = []
        total_loss = 0.0
        gather_x = []

        for i, sfeat in enumerate(sub_features):
            if sfeat.shape[1] == 0:
                continue
            loss_i, adj_i, usage_i, codebook_i = self.get_adj_list[i](sfeat)
            x_i = self.at(self.res_gcn_list[i](sfeat, adj_i))

            usage_list.append(usage_i)
            codebook_list.append(codebook_i)
            total_loss += loss_i
            gather_x.append(x_i)

        if len(gather_x) == 0:
            return total_loss, None, usage_list, codebook_list

        region_x = torch.concat(gather_x, dim=1)
        loss_co, adj_co, usage_co, codebook_co = self.get_adj_co(self.att_coarsen(region_x, self.weight1))
        x_co = self.res_gcn_co(self.att_coarsen(region_x, self.weight1), adj_co)

        usage_list.append(usage_co)
        codebook_list.append(codebook_co)
        total_loss += 2.0 * loss_co

        return total_loss, x_co, usage_list, codebook_list

    def att_coarsen(self, features, weight):
        feature_with_weight = torch.einsum('b n d, d h -> b n h', features, weight)
        feature_T = rearrange(feature_with_weight, 'b n h -> b h n')
        att_weight_matrix = torch.einsum('b n h, b h m -> b n m', feature_with_weight, feature_T)
        att_weight_vector = torch.sum(att_weight_matrix, dim=2)
        att_vec = self.att_softmax(att_weight_vector)
        sub_feature_ = torch.einsum('b n, b n d -> b d', att_vec, features)
        coarsen_x = rearrange(sub_feature_, 'b d -> b 1 d')
        return coarsen_x

    def region_x(self, x):
        sub_feature_list = []
        for c_idx in range(self.subgraph_num):
            ch_mask = (self.cluster_labels == c_idx)
            sub_feature_list.append(x[:, ch_mask, :])
        return sub_feature_list