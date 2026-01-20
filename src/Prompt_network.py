
from functools import partial

import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn import GATConv

import copy

# sinusoidal positional embeds

##############----todo--扩散卷积一
class DiffusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=2):
        super(DiffusionConv, self).__init__()
        self.k = k  # 扩散阶数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.theta = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_channels, out_channels))
            for _ in range(k)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.theta:
            nn.init.xavier_uniform_(param)

    def forward(self, x, edge_index, edge_weight=None):
        '''
        x: [N, in_channels]
        edge_index: [2, E]
        edge_weight: [E] or None
        '''
        N = x.size(0)
        out = torch.zeros(N, self.out_channels, device=x.device)

        # 构造邻接矩阵
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, (N, N))
        adj = torch.sparse.softmax(adj, dim=1)  # 归一化（转移概率）

        x_k = x  # x^(0)
        for i in range(self.k):
            x_k = torch.sparse.mm(adj, x_k)  # diffusion propagation
            out += x_k @ self.theta[i]

        return out


class DCRNNWrapper(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k=2):
        super(DCRNNWrapper, self).__init__()
        self.diff1 = DiffusionConv(in_channels, hidden_channels, k)
        self.relu = nn.ReLU()
        self.diff2 = DiffusionConv(hidden_channels, out_channels, k)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.diff1(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.diff2(x, edge_index, edge_weight)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels // heads, heads=heads)
        self.gat2 = GATConv(hidden_channels, out_channels, heads=1)  # 输出只需一个头

    def forward(self, x, edge_index, edge_att=None):  # edge_att 可忽略或用于边特征注意力扩展
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        return x




class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        print(in_channels, hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_att=None):
        x = self.conv1(x, edge_index, edge_att)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_att)
        return x





# class MultiHeadCrossAttention(nn.Module):
#     def __init__(self, dim, num_heads):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
#
#         self.q_proj = nn.Linear(dim, dim)
#         self.k_proj = nn.Linear(dim, dim)
#         self.v_proj = nn.Linear(dim, dim)
#         self.out_proj = nn.Linear(dim, dim)
#         self.scale = self.head_dim ** -0.5
#
#     def forward(self, q, kv):
#         B, N, C = q.shape
#         q = self.q_proj(q).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B, H, N, D
#         k = self.k_proj(kv).view(B, kv.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
#         v = self.v_proj(kv).view(B, kv.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
#
#         attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # B, H, N, M
#         attn = attn_scores.softmax(dim=-1)
#         out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 合并头
#
#         return self.out_proj(out), attn

####双路注意力增强记忆模块（Dual-Attention Enhanced Memory, DAEM）   ##无作用
# class Memory2(nn.Module):
#     """ Memory prompt
#     """
#
#     def __init__(self, num_memory, memory_dim, temporal_agg=False, args=None):
#         super().__init__()
#
#         self.args = args
#
#         self.num_memory = num_memory
#         self.memory_dim = memory_dim
#         self.temporal_agg = temporal_agg
#
#         self.memMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # M,C
#         self.keyMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # M,C
#
#         self.memMatrix.requires_grad = True
#         self.keyMatrix.requires_grad = True
#
#         self.x_proj = nn.Linear(memory_dim, memory_dim)
#         self.cross_attn = MultiHeadCrossAttention(dim=memory_dim, num_heads=4)  # 你可以调头数
#         self.gate_linear = nn.Linear(memory_dim * 2, memory_dim)  # 门控融合
#
#
#         if temporal_agg:
#             encdoer_layer = nn.TransformerEncoderLayer(d_model=memory_dim, nhead=4, dim_feedforward=memory_dim,
#                                                        batch_first=True)
#             self.encoder = nn.TransformerEncoder(encoder_layer=encdoer_layer, num_layers=1)
#
#         self.initialize_weights()
#
#         print("model initialized memory")
#
#     def initialize_weights(self):
#         torch.nn.init.trunc_normal_(self.memMatrix, std=0.02)
#         torch.nn.init.trunc_normal_(self.keyMatrix, std=0.02)
#
#         # initialize nn.Linear and nn.LayerNorm
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def forward(self, x, Type='', shape=None,cross_input=None):
#         """
#         :param x: query features with size [N,C], where N is the number of query items,
#                   C is same as dimension of memory slot
#
#         :return: query output retrieved from memory, with the same size as x.
#         """
#         # dot product
#
#         if self.temporal_agg:
#             x = self.encoder(x).mean(dim=1)
#
#         assert x.shape[-1] == self.memMatrix.shape[-1] == self.keyMatrix.shape[-1], "dimension mismatch"
#         # 生成查询向量
#         x_query = torch.tanh(self.x_proj(x))
#         # 生成
#         att_weight = F.linear(input=x_query, weight=self.keyMatrix)  # [N,C] by [M,C]^T --> [N,M]
#
#         att_weight = F.softmax(att_weight, dim=-1)  # NxM
#
#         out_memory = F.linear(att_weight, self.memMatrix.permute(1, 0))  # [N,M] by [M,C]  --> [N,C]
#
#         # 支持多头交叉注意力
#
#         if cross_input.ndim == 2:
#             cross_input = cross_input.unsqueeze(0)  # reshape 成 [1, N, C] 形式
#         if x.ndim == 2:
#             x = x.unsqueeze(0)
#
#         cross_out, _ = self.cross_attn(x, cross_input)
#
#         # 门控融合
#         gate = torch.sigmoid(self.gate_linear(torch.cat([out_memory, cross_out.squeeze(0)], dim=-1)))
#         out_fused = gate * out_memory + (1 - gate) * cross_out.squeeze(0)
#
#
#         return dict(out=out_fused, att_weight=att_weight)


class MultiScaleMemory(nn.Module):
    """ 多尺度 Memory Prompt 模块 """

    def __init__(self, num_memory, memory_dim, n_scales=2, temporal_agg=False, args=None):
        super().__init__()
        self.n_scales = n_scales
        self.memory_dim = memory_dim
        self.temporal_agg = temporal_agg
        self.args = args

        # 多尺度记忆与键向量
        self.memMatrix_list = nn.ParameterList([
            nn.Parameter(torch.zeros(num_memory, memory_dim)) for _ in range(n_scales)
        ])
        self.keyMatrix_list = nn.ParameterList([
            nn.Parameter(torch.zeros(num_memory, memory_dim)) for _ in range(n_scales)
        ])

        self.x_proj = nn.Linear(memory_dim, memory_dim)

        if temporal_agg:
            encoder_layer = nn.TransformerEncoderLayer(d_model=memory_dim, nhead=4,
                                                        dim_feedforward=memory_dim, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=1)

        self._initialize_weights()
        print("model initialized memory")

    def _initialize_weights(self):
        for i in range(self.n_scales):
            nn.init.trunc_normal_(self.memMatrix_list[i], std=0.02)
            nn.init.trunc_normal_(self.keyMatrix_list[i], std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, Type='', shape=None):
        """
        :param x: [N, C]
        :return: {'out': [N, C], 'att_weight': [N, M]}
        """
        if self.temporal_agg:
            x = self.encoder(x).mean(dim=1)

        x_query = torch.tanh(self.x_proj(x))  # [N, C]

        outputs = []
        weights = []

        for i in range(self.n_scales):
            keyMatrix = self.keyMatrix_list[i]     # [M, C]
            memMatrix = self.memMatrix_list[i]     # [M, C]

            att = F.linear(x_query, keyMatrix)     # [N, M]
            att = F.softmax(att, dim=-1)           # [N, M]
            out = F.linear(att, memMatrix.T)       # [N, C]

            outputs.append(out)
            weights.append(att)

        # 多尺度融合（加权平均）
        all_out = torch.stack(outputs, dim=1)      # [N, S, C]
        scale_weights = F.softmax(torch.mean(all_out, dim=-1), dim=1)  # [N, S]
        out_fused = torch.sum(all_out * scale_weights.unsqueeze(-1), dim=1)  # [N, C]

        # return {'out': out_fused, 'att_weight': weights}
        return dict(out=out_fused, att_weight=weights)





class Memory(nn.Module):
    """ Memory prompt
    """
    def __init__(self, num_memory, memory_dim, temporal_agg = False, args=None):
        super().__init__()

        self.args = args

        self.num_memory = num_memory
        self.memory_dim = memory_dim
        self.temporal_agg = temporal_agg

        self.memMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # M,C
        self.keyMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # M,C

        self.memMatrix.requires_grad = True
        self.keyMatrix.requires_grad = True

        self.x_proj = nn.Linear(memory_dim, memory_dim)

        if temporal_agg:
            encdoer_layer = nn.TransformerEncoderLayer(d_model=memory_dim, nhead=4, dim_feedforward=memory_dim, batch_first = True)
            self.encoder = nn.TransformerEncoder(encoder_layer=encdoer_layer, num_layers=1)
        
        self.initialize_weights()

        print('num memory = ', num_memory)
        print("model initialized memory")


    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.memMatrix, std=0.02)
        torch.nn.init.trunc_normal_(self.keyMatrix, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,x,Type='',shape=None):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """
        # dot product

        if self.temporal_agg:
            x = self.encoder(x).mean(dim=1)

        assert x.shape[-1]==self.memMatrix.shape[-1]==self.keyMatrix.shape[-1], "dimension mismatch"
        #生成查询向量
        x_query = torch.tanh(self.x_proj(x))
        #生成
        att_weight = F.linear(input=x_query, weight=self.keyMatrix)  # [N,C] by [M,C]^T --> [N,M]

        att_weight = F.softmax(att_weight, dim=-1)  # NxM

        out = F.linear(att_weight, self.memMatrix.permute(1, 0))  # [N,M] by [M,C]  --> [N,C]

        return dict(out=out, att_weight=att_weight)



# class DynamicMemory(nn.Module):
#     def __init__(self, num_memory, memory_dim, temporal_agg=False, args=None):
#         super().__init__()
#
#         self.num_memory = num_memory
#         self.memory_dim = memory_dim
#         self.temporal_agg = temporal_agg
#         self.memMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # 存储记忆的矩阵
#         self.keyMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # 存储记忆键的矩阵
#
#         self.x_proj = nn.Linear(memory_dim, memory_dim)
#         if temporal_agg:
#             encoder_layer = nn.TransformerEncoderLayer(d_model=memory_dim, nhead=4, dim_feedforward=memory_dim, batch_first=True)
#             self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=1)
#
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         # 初始化记忆矩阵和键矩阵
#         torch.nn.init.trunc_normal_(self.memMatrix, std=0.02)
#         torch.nn.init.trunc_normal_(self.keyMatrix, std=0.02)
#
#         # 初始化线性层
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def forward(self, x):
#         # x: 当前的输入数据，形状为 [N, C]，N是批量大小，C是特征数
#         x_query = torch.tanh(self.x_proj(x))  # 生成查询向量
#
#         # 计算当前时间步的相关性分数（基于记忆键矩阵）
#         att_weight = F.linear(x_query, self.keyMatrix)  # 计算注意力权重，形状为 [N, M]
#         att_weight = F.softmax(att_weight, dim=-1)  # 使用softmax归一化
#
#         # 根据注意力权重从记忆矩阵中检索信息
#         out = F.linear(att_weight, self.memMatrix.permute(1, 0))  # 形状为 [N, C]
#
#         self.memory_dim = memory + att_weight.unsqueeze(-1) * x
#
#         return out, att_weight


