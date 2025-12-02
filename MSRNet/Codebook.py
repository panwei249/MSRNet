import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class DynamicCodebook(nn.Module):
    """
    动态码本，用于在训练过程中对VQ码字进行自适应更新。
    """
    def __init__(self, in_chan, num_embeddings, input_chan, embedding_dim, commitment_cost,
                 usage_threshold=1e-3):
        """
        :param in_chan: 只是为了保持原接口一致 (不一定用得到)
        :param num_embeddings: 初始的码字数
        :param input_chan: 输入特征的某个维度 (和你原先的 input_dim 匹配)
        :param embedding_dim: 每个码字的嵌入维度
        :param commitment_cost: vq 的损失系数
        :param usage_threshold: 使用率阈值，若某码字 usage < threshold，则将其视为“未使用”
        """
        super(DynamicCodebook, self).__init__()
        self._vq = DynamicVectorQuantizer(
            num_embeddings=num_embeddings,
            input_dim=input_chan**2,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            usage_threshold=usage_threshold
        )

    def forward(self, x):
        # 直接调用新的动态向量量化器
        loss, x_recon, usage, codebook = self._vq(x)
        return loss, x_recon, usage, codebook


class DynamicVectorQuantizer(nn.Module):
    """
    与原先的 VectorQuantizer 类似，但在每次 forward 后执行一次简单的“动态更新”：
      - 若某个码字在本次 batch 的使用率 < usage_threshold， 则随机重置该码字（相当于让它有机会学习新的特征）。
    你也可以改成更复杂的合并/拆分等自适应策略。
    """
    def __init__(self, num_embeddings, input_dim, embedding_dim, commitment_cost,
                 usage_threshold=1e-3):
        super(DynamicVectorQuantizer, self).__init__()

        self.running_usage = nn.Parameter(
            torch.zeros(num_embeddings), requires_grad=False
        )
        self.alpha = 0.9  # 衰减系数
        self.high_usage_threshold = 0.1  # 用于“克隆”参考的高使用率阈值

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        self.input_dim = input_dim
        self.usage_threshold = usage_threshold

        # 初始化embedding参数
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)

        # 线性转换，用于把 input_dim -> embedding_dim / embedding_dim -> input_dim
        self.linear1 = nn.Linear(self.input_dim, self._embedding_dim)
        self.linear2 = nn.Linear(self._embedding_dim, self.input_dim)

        # 两个BN，用于稳定训练
        self.bn1 = nn.BatchNorm1d(self.input_dim)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

        self.bn2 = nn.BatchNorm1d(self._embedding_dim)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

        self.at = nn.SELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        """
        :param inputs: 形状 [..., input_dim]，通常是 [B*N, input_dim]后再reshape
        :return:
            loss: VQ 损失
            quantized: 量化后的张量，形状与 inputs 相同
            usage: 当前 batch 每个码字被用到的平均值
            codebook: 当前码字表
        """
        input_shape = inputs.shape
        # 1) Flatten input => [batch_size*N, input_dim]
        flat_input = inputs.reshape(-1, self.input_dim)

        # 2) BN + Dropout + Linear => 先把原空间映射到 embedding_dim
        flat_input = self.dropout(self.bn1(flat_input))
        flat_input = self.linear1(flat_input)

        # 3) 计算与每个码字的距离
        #   embedding.weight: [num_embeddings, embedding_dim]
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # 4) 找到最小距离对应的索引 => 对应 one-hot
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0],
                                self._num_embeddings,
                                device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 5) 根据编码索引得到量化后的向量 quantized
        quantized = torch.matmul(encodings, self._embedding.weight)
        # BN + Dropout + Linear => 投回 input_dim
        quantized = self.dropout(self.bn2(quantized))
        quantized = self.linear2(quantized).reshape(input_shape)

        # 6) 计算 VQ Loss
        #  e_latent_loss = mse(quantized.detach(), inputs)
        #  q_latent_loss = mse(quantized, inputs.detach())
        #  loss = q_latent_loss + commitment_cost*e_latent_loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # 7) Straight-through: (quantized - inputs).detach()
        quantized = inputs + (quantized - inputs).detach()

        # 8) 统计 usage
        usage = torch.mean(encodings, dim=0).tolist()  # [num_embeddings]

        # 9) 动态更新 codebook
        self._update_codebook(usage, inputs.device)

        # 10) 取出当前 codebook（仅做日志或可视化用）
        codebook = self._embedding.weight.data.cpu().numpy().tolist()

        return loss, quantized, usage, codebook

    def _update_codebook(self, usage_batch, device):
        """
        usage_batch: 当前 batch 计算的 usage (列表或张量)
        实现功能：
          1) running usage 的指数滑动平均
          2) 如果 usage < usage_threshold => 尝试替换为高 usage 码字 + 噪声
        """

        usage_batch_t = torch.tensor(usage_batch, device=device, dtype=torch.float32)

        # (A) 更新滑动平均 usage
        with torch.no_grad():
            # 历史衰减
            self.running_usage *= self.alpha
            # 融合当前 batch 统计
            self.running_usage += (1.0 - self.alpha) * usage_batch_t

        # 取平滑后的 usage 值
        current_usage = self.running_usage.clone().detach()

        # (B) 找到低使用率的码字索引
        rare_indices = (current_usage < self.usage_threshold).nonzero(as_tuple=True)[0]
        if len(rare_indices) == 0:
            return  # 都高于 threshold，无需更新

        # (C) 找到高使用率码字索引
        high_usage_indices = (current_usage > self.high_usage_threshold).nonzero(as_tuple=True)[0]
        if len(high_usage_indices) == 0:
            # 若不存在高 usage 码字，则只能随机重置低 usage 码字
            for idx in rare_indices:
                nn.init.uniform_(
                    self._embedding.weight[idx],
                    -1 / self._num_embeddings,
                    1 / self._num_embeddings
                )
            return

        # (D) 克隆替换：
        #  将低使用率码字替换为 “随机选出的高使用率码字 + 少量噪声”
        with torch.no_grad():
            for idx in rare_indices:
                # donor_idx_t 是形如 [1] 的张量 => 先变成标量
                donor_idx_t = high_usage_indices[torch.randint(len(high_usage_indices), (1,))]
                donor_idx = donor_idx_t.item()  # 转为 int

                # 取出高 usage 码字向量
                donor_vec = self._embedding.weight[donor_idx]
                # 加上少量噪声
                noise = 0.05 * torch.randn_like(donor_vec)
                new_vec = donor_vec + noise
                # 覆盖低 usage 码字
                self._embedding.weight[idx].copy_(new_vec)
