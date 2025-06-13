import time
from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import numpy as np
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import DropPath, Mlp, Attention

from Embed import GraphEmbedding, DataEmbedding, TokenEmbedding, SpatialPatchEmb, get_2d_sincos_pos_embed, \
    get_2d_sincos_pos_embed_with_resolution, get_1d_sincos_pos_embed_from_grid, \
    get_1d_sincos_pos_embed_from_grid_with_resolution
from mask_strategy import *
import copy
from Autoformer_EncDec import  series_decomp
from Prompt_network import Memory, GCN,GAT,DCRNNWrapper


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index




class TransformerDecoderModel(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerDecoderModel, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        self.linear = nn.Linear(d_model, d_model)  # Adjust the output dimension as needed

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask,
                                          memory_key_padding_mask)
        output = self.linear(output)
        return output


def model_select(args, **kwargs):
    if args.size == 'small':
        model = FedFlow_model(
            embed_dim=128,
            depth=4,
            decoder_embed_dim=128,
            decoder_depth=4,
            num_heads=4,
            decoder_num_heads=4,
            mlp_ratio=2,
            t_patch_size=args.t_patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb=args.pos_emb,
            no_qkv_bias=bool(args.no_qkv_bias),
            args=args,
            **kwargs,
        )
        return model

    elif args.size == 'middle':
        model = FedFlow_model(
            embed_dim=256,
            depth=4,
            decoder_embed_dim=256,
            decoder_depth=4,
            num_heads=4,
            decoder_num_heads=4,
            mlp_ratio=2,
            t_patch_size=args.t_patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb=args.pos_emb,
            no_qkv_bias=bool(args.no_qkv_bias),
            args=args,
            **kwargs,
        )
        return model

    elif args.size == 'large':
        model = FedFlow_model(
            embed_dim=256,
            depth=6,
            decoder_embed_dim=256,
            decoder_depth=6,
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=2,
            t_patch_size=args.t_patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb=args.pos_emb,
            no_qkv_bias=bool(args.no_qkv_bias),
            args=args,
            **kwargs,
        )
        return model

#
# class Attention(nn.Module):
#     def __init__(
#             self,
#             dim,  # 输入特征维度
#             num_heads=8,  # 注意力头数
#             qkv_bias=False,  # 是否在查询（Q）、键（K）、值（V）的线性变换中使用偏置
#             qk_scale=None,  # 查询和键的点积缩放因子 默认为 head_dim**-0.5
#             attn_drop=0.0,  # 注意力得分的dropout率
#             proj_drop=0.0,  # 输出投影的dropout率
#             input_size=(4, 14, 14),  # 输入张量的形状
#     ):
#         super().__init__()
#         # assert dim % num_heads == 0, "dim should be divisible by num_heads"  # 确保 dim 可以被 num_heads 整除，以便每个头有相同的维度。
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v = nn.Linear(dim, dim, bias=qkv_bias)
#
#         # assert attn_drop == 0.0  # do not use
#         self.proj = nn.Linear(dim, dim, bias=qkv_bias)  # 定义一个线性层 self.proj 用于输出投影
#         self.proj_drop = nn.Dropout(proj_drop)  # dropout层
#         self.input_size = input_size
#         # assert input_size[1] == input_size[2]
#
#
#       #输入：[51, 768, 256]
#     def forward(self, x, attn_bias={}):
#
#
#         B, N, C = x.shape
#         q = (
#             self.q(x)
#             .reshape(B, N, self.num_heads, C // self.num_heads)
#             .permute(0, 2, 1, 3)
#         )
#         k = (
#             self.k(x)
#             .reshape(B, N, self.num_heads, C // self.num_heads)
#             .permute(0, 2, 1, 3)
#         )
#         v = (
#             self.v(x)
#             .reshape(B, N, self.num_heads, C // self.num_heads)
#             .permute(0, 2, 1, 3)
#         )
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # self.scale 缩放点积结果
#
#
#         if attn_bias != {}:
#
#             if 'bias_t' in attn_bias:
#                 T = attn.shape[-1] // attn_bias['bias_t'].shape[-1]
#             elif 'bias_f' in attn_bias:
#                 T = attn.shape[-1] // attn_bias['bias_f'].shape[-1]
#             elif 'topo' in attn_bias:
#                 T = attn.shape[-1] // attn_bias['topo'].shape[-1]
#
#             if 'bias_t' in attn_bias:
#                 attn_bias_t = attn_bias['bias_t'].unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=4)
#                 attn_bias_t = attn_bias_t.repeat(1, self.num_heads, T, 1, T, 1)
#                 attn_bias_t = attn_bias_t.reshape(attn_bias_t.shape[0], self.num_heads, attn.shape[-2], attn.shape[-1])
#
#                 # assert attn.shape == attn_bias_t.shape
#
#                 attn += attn_bias_t
#
#             if 'bias_f' in attn_bias:
#                 attn_bias_f = attn_bias['bias_f'].unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=4)
#                 attn_bias_f = attn_bias_f.repeat(1, self.num_heads, T, 1, T, 1)
#                 attn_bias_f = attn_bias_f.reshape(attn_bias_f.shape[0], self.num_heads, attn.shape[-2], attn.shape[-1])
#
#                 # assert attn.shape == attn_bias_f.shape
#
#                 attn += attn_bias_f
#
#             if 'topo' in attn_bias:
#                 attn_bias_topo = attn_bias['topo'].unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=4)
#                 attn_bias_topo = attn_bias_topo.repeat(1, self.num_heads, T, 1, T, 1)
#                 attn_bias_topo = attn_bias_topo.reshape(attn_bias_topo.shape[0], self.num_heads, attn.shape[-2],
#                                                         attn.shape[-1])
#
#                 # assert attn.shape == attn_bias_topo.shape
#
#                 attn += attn_bias_topo
#
#         attn = attn.softmax(dim=-1)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         x = x.view(B, -1, C)
#
#         return x




class AutoCorrelationWrapper(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
        factor=1,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.dim = dim

        # 实例化 AutoCorrelation 模块
        auto_corr = AutoCorrelation(
            factor=factor,
            scale=qk_scale,
            attention_dropout=attention_dropout,
            output_attention=output_attention
        )
        self.auto_corr_layer = AutoCorrelationLayer(
            correlation=auto_corr,
            d_model=dim,
            n_heads=num_heads
        )

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias={}):
        # x: [B, N, C] -> [B, L, C]
        B, N, C = x.shape

        # AutoCorrelationLayer 要求 queries, keys, values 分别为 [B, L, C]
        out, attn = self.auto_corr_layer(x, x, x, attn_mask=None)

        out = self.proj_drop(out)

        return out



class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False, configs=None):
        super(AutoCorrelation, self).__init__()
        print('Autocorrelation used !')
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.agg = None
        # self.use_wavelet = configs.wavelet

    # @decor_time
    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg  # size=[B, H, d, S]

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, L, -1)
        return self.out_projection(out), attn


# def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
#     """
#     get modes on frequency domain:
#     'random' means sampling randomly;
#     'else' means sampling the lowest modes;
#     """
#     modes = min(modes, seq_len // 2)
#     if mode_select_method == 'random':
#         index = list(range(0, seq_len // 2))
#         np.random.shuffle(index)
#         index = index[:modes]
#     else:
#         index = list(range(0, modes))
#     index.sort()
#     return index


# ########## fourier layer #############
class FourierBlock(nn.Module):
    def __init__(self, num_heads,in_channels, out_channels, seq_len, modes=0, mode_select_method='random',d_keys=None,
                 d_values=None):
        super(FourierBlock, self).__init__()
        d_keys = d_keys or (in_channels // num_heads)
        d_values = d_values or (in_channels // num_heads)

        print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        # get modes on frequency domain
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        print('modes={}, index={}'.format(modes, self.index))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(4, in_channels // 4, out_channels // 4, len(self.index), dtype=torch.cfloat))
        self.num_heads=num_heads
        self.query_projection = nn.Linear(in_channels, d_keys * num_heads)
        self.key_projection = nn.Linear(in_channels, d_keys * num_heads)
        self.value_projection = nn.Linear(in_channels, d_values * num_heads)


    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, x, mask):

        #print(x.shape)   #[8, 6, 256]    [6, 315, 256]

        queries=x
        keys=x
        values =x
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.num_heads

        queries = self.query_projection(queries).view(B, L, H, -1)#[25, 200, 4, 64]
        # keys = self.key_projection(keys).view(B, S, H, -1)
        # values = self.value_projection(values).view(B, S, H, -1)


        # size = [B, L, H, E]
        B, L, H, E = queries.shape
        x = queries.permute(0, 2, 3, 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        #print(x_ft.shape)     #[8, 4, 64, 4]     [6, 4, 64, 158]

        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)

        #print(out_ft.shape)   #[8, 4, 64, 4]   [6, 4, 64, 158]


        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))

        out = x.view(B, L, -1)
        return out


# ########## Fourier Cross Former ####################
class FourierCrossAttention(nn.Module):
    def __init__(self,num_heads,in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random',
                 activation='tanh', policy=0,d_keys=None,d_values=None):
        super(FourierCrossAttention, self).__init__()
        d_keys = d_keys or (in_channels // num_heads)
        d_values = d_values or (in_channels // num_heads)

        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        # get modes for queries and keys (& values) on frequency domain
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(4, in_channels // 4, out_channels // 4, len(self.index_q), dtype=torch.cfloat))

        self.query_projection = nn.Linear(in_channels, d_keys * num_heads)
        self.key_projection = nn.Linear(in_channels, d_keys * num_heads)
        self.value_projection = nn.Linear(in_channels, d_values * num_heads)
        self.num_heads=num_heads


    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, x, mask):

        queries = x
        keys = x
        values = x
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.num_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # size = [B, L, H, E]
        B, L, H, E = queries.shape
        xq = queries.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = keys.permute(0, 2, 3, 1)
        xv = values.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # perform attention mechanism on frequency domain
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))

        out = x.view(B, L, -1)
        return out



# class FourierBlockWrapper(nn.Module):
#     def __init__(self, dim, num_heads=8, input_size=(4, 14, 14), modes=64, mode_select_method='random'):
#         super(FourierBlockWrapper, self).__init__()
#         self.num_heads = num_heads
#         self.dim = dim
#         self.input_size = input_size
#         self.seq_len = input_size[1] * input_size[2]
#         self.head_dim = dim // num_heads
#
#         self.qkv = nn.Linear(dim, dim * 3)  # 生成q, k, v
#         self.self_att = FourierBlock(in_channels=configs.d_model,
#                                         out_channels=configs.d_model,
#                                         seq_len=self.seq_len,
#                                         modes=configs.modes,
#                                         mode_select_method=configs.mode_select)
#         self.proj = nn.Linear(dim, dim)
#
#     def forward(self, x, attn_bias={}):
#
#
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # shape: [B, N, H, D]
#         q = q.permute(0, 2, 1, 3)  # [B, H, N, D]
#         k = k.permute(0, 2, 1, 3)
#         v = v.permute(0, 2, 1, 3)
#
#         out, _ = self.fourier(q, k, v, None)  # FourierBlock 忽略 attn_bias
#         out = out.permute(0, 2, 1, 3).reshape(B, N, C)
#         out = self.proj(out)
#         return out
#


# class FourierCrossAttentionWrapper(nn.Module):
#     def __init__(self, dim, num_heads=8, input_size=(4, 14, 14), modes=64, mode_select_method='random', activation='tanh'):
#         super(FourierCrossAttentionWrapper, self).__init__()
#         self.num_heads = num_heads
#         self.dim = dim
#         self.input_size = input_size
#         self.seq_len = input_size[1] * input_size[2]
#         self.head_dim = dim // num_heads
#
#         self.q = nn.Linear(dim, dim)
#         self.k = nn.Linear(dim, dim)
#         self.v = nn.Linear(dim, dim)
#
#         self.fourier_cross = FourierCrossAttention(
#             in_channels=self.head_dim,
#             out_channels=self.head_dim,
#             seq_len_q=self.seq_len,
#             seq_len_kv=self.seq_len,
#             modes=modes,
#             mode_select_method=mode_select_method,
#             activation=activation
#         )
#         self.proj = nn.Linear(dim, dim)
#
#     def forward(self, x, attn_bias={}):
#         B, N, C = x.shape
#         q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         out, _ = self.fourier_cross(q, k, v, None)
#         out = out.permute(0, 2, 1, 3).reshape(B, N, C)
#         out = self.proj(out)
#         return out
#
#





class EncoderBlock(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,  # MLP 隐藏层维度与输入维度的比例，默认为 4.0
            qkv_bias=False,
            qk_scale=None,
            drop=0.05,  # MLP 中的 dropout 概率 0
            attn_drop=0.01,
            drop_path=0.05,  #0
            act_layer=nn.GELU,  # 激活函数，默认为 GELU
            norm_layer=nn.LayerNorm,
            # cross_attention=AutoCorrelationWrapper,  # 自定义的注意力机制类，默认为 Attention
            self_attention=FourierBlock

    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = self_attention(
        #     dim,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     attn_drop=attn_drop,
        #     proj_drop=drop,
        # )

        self.attn = self_attention(num_heads=num_heads,
                                 in_channels=256,
                                        out_channels=256,
                                        seq_len=12,
                                        modes=64,
                                        mode_select_method='random')

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,  # 隐藏层特征维度
            act_layer=act_layer,  # 激活函数
            drop=drop,
        )
        moving_avg = 24

        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.activation = F.gelu
        self.dropout = nn.Dropout(0.05)
        d_model=512
        d_ff=2028



    def forward(self, x, attn_bias={}):
        #输入输出：[51, 768, 256]

        # new_x = x + self.drop_path(self.attn(self.norm1(x), attn_bias=attn_bias))
        new_x = x + self.drop_path(self.attn(x,None))

        x,_ = self.decomp1(new_x)  # 趋势分解
        y=x

        #加一层激活函数
        y = y + self.drop_path(self.activation(self.mlp(self.norm2(y))))

        res,_ = self.decomp2(x+y)

        #seasonal_init, trend_init = self.decomp(x)   #趋势分解
        #输出[51, 768, 256]

        return res



class DecoderBlock(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,  # MLP 隐藏层维度与输入维度的比例，默认为 4.0
            qkv_bias=False,
            qk_scale=None,
            drop=0.05,  # MLP 中的 dropout 概率
            attn_drop=0.01,
            drop_path=0.05,
            act_layer=nn.GELU,  # 激活函数，默认为 GELU
            norm_layer=nn.LayerNorm,
            cross_attention=FourierCrossAttention,  # 自定义的注意力机制类，默认为 Attention
            self_attention=FourierBlock

    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        # self.attn = self_attention(
        #     dim,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     attn_drop=attn_drop,
        #     proj_drop=drop,
        # )
        #
        # self.correlation = cross_attention(
        #     dim,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     attn_drop=attn_drop,
        #     proj_drop=drop,
        # )

        self.seq_len=12
        self.pred_len=12

        self.attn = self_attention(num_heads=num_heads,
                                 in_channels=256,
                                        out_channels=256,
                                        seq_len=self.seq_len // 2 + self.pred_len,
                                        modes=64,
                                        mode_select_method='random')
        self.correlation = cross_attention(num_heads=num_heads,
                                           in_channels=256,
                                                  out_channels=256,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=64,
                                                  mode_select_method='random')



        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,  # 隐藏层特征维度
            act_layer=act_layer,  # 激活函数
            drop=drop,
        )
        self.moving_avg = 24

        self.decomp1 = series_decomp(self.moving_avg)
        self.decomp2 = series_decomp(self.moving_avg)
        self.decomp3 = series_decomp(self.moving_avg)
        self.activation = F.gelu
        self.dropout=0.05


    def forward(self, x, attn_bias={}):
        #输入输出：[51, 768, 256]

        x = x + self.drop_path(self.attn(self.norm1(x), None))
        x, trend1 = self.decomp1(x)
        x = x + self.drop_path(self.correlation(self.norm3(x), None))
        x, trend2 = self.decomp2(x)

        y=x
        #加一层激活函数
        y = x + self.drop_path(self.activation(self.mlp(self.norm2(y))))
        x,trend3 = self.decomp2(x+y)

        residual_trend = trend1 + trend2 + trend3

        #residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)

        #seasonal_init, trend_init = self.decomp(x)   #趋势分解
        #输出[51, 768, 256]

        return x, residual_trend







# class Block(nn.Module):
#     """
#     Transformer Block with specified Attention function
#     """
#
#     def __init__(
#             self,
#             dim,
#             num_heads,
#             mlp_ratio=4.0,  # MLP 隐藏层维度与输入维度的比例，默认为 4.0
#             qkv_bias=False,
#             qk_scale=None,
#             drop=0.0,  # MLP 中的 dropout 概率
#             attn_drop=0.0,
#             drop_path=0.0,
#             act_layer=nn.GELU,  # 激活函数，默认为 GELU
#             norm_layer=nn.LayerNorm,
#             attn_func=AutoCorrelationWrapper,  # 自定义的注意力机制类，默认为 Attention
#     ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = attn_func(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=drop,
#         )
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(
#             in_features=dim,
#             hidden_features=mlp_hidden_dim,  # 隐藏层特征维度
#             act_layer=act_layer,  # 激活函数
#             drop=drop,
#         )
#         self.kernel_size = 24
#
#         self.decomp = series_decomp(self.kernel_size)
#
#
#
#     def forward(self, x, attn_bias={}):
#         #输入输出：[51, 768, 256]
#
#         x = x + self.drop_path(self.attn(self.norm1(x), attn_bias=attn_bias))
#
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         #seasonal_init, trend_init = self.decomp(x)   #趋势分解
#         #输出[51, 768, 256]
#
#         return x






class FedFlow_model(nn.Module):
    def __init__(self, in_chans=1,  # 24??
                 embed_dim=1024, decoder_embed_dim=512, depth=4, decoder_depth=4, num_heads=16, decoder_num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, t_patch_size=1,
                 no_qkv_bias=False, pos_emb='trivial', args=None, ):
        super().__init__()

        self.args = args

        self.pos_emb = pos_emb

        self.Embedding_patch = DataEmbedding(1, embed_dim, args=self.args)
        self.Embedding_patch_graph = GraphEmbedding(1, embed_dim, GridEmb=self.Embedding_patch, args=self.args)

        # mask

        self.t_patch_size = t_patch_size
        self.decoder_embed_dim = decoder_embed_dim
        self.in_chans = in_chans

        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 1024, embed_dim)
        )
        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, 50, embed_dim)
        )

        self.decoder_pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 1024, decoder_embed_dim)
        )
        self.decoder_pos_embed_temporal = nn.Parameter(
            torch.zeros(1, 50, decoder_embed_dim)
        )
        # self.blocks = nn.ModuleList(  ##编码器
        #     [
        #         Block(
        #             embed_dim,
        #             num_heads,
        #             mlp_ratio,
        #             qkv_bias=not no_qkv_bias,
        #             qk_scale=None,
        #             norm_layer=norm_layer,
        #         )
        #         for i in range(depth)
        #     ]
        # )

        self.enblocks = nn.ModuleList(  ##编码器
            [
                EncoderBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.deblocks = nn.ModuleList(  ##编码器
            [
                DecoderBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.no_qkv_bias = no_qkv_bias
        self.norm_layer = norm_layer

        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=not self.args.no_qkv_bias)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        #
        # self.decoder_blocks = nn.ModuleList(  ##解码器
        #     [
        #         Block(
        #             decoder_embed_dim,
        #             decoder_num_heads,
        #             mlp_ratio,
        #             qkv_bias=not no_qkv_bias,
        #             qk_scale=None,
        #             norm_layer=norm_layer,
        #         )
        #         for i in range(decoder_depth)
        #     ]
        # )

        encdoer_layer2 = nn.TransformerEncoderLayer(d_model=decoder_embed_dim, nhead=2,
                                                    dim_feedforward=decoder_embed_dim // 2, batch_first=True)
        self.spatial_attn_spec_tmp = nn.TransformerEncoder(encoder_layer=encdoer_layer2, num_layers=1)

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.pred_model = TransformerDecoderModel(d_model=decoder_embed_dim, dim_feedforward=decoder_embed_dim // 2,
                                                  nhead=2, num_decoder_layers=1)
        self.pred_model_linear_GraphBJ = nn.Linear(decoder_embed_dim, self.t_patch_size * 105 * in_chans)
        self.pred_model_linear_GraphNJ = nn.Linear(decoder_embed_dim, self.t_patch_size * 105 * in_chans)
        self.pred_model_linear_GraphSH = nn.Linear(decoder_embed_dim, self.t_patch_size * 210 * in_chans)
        self.pred_model_linear_GraphPEMS = nn.Linear(decoder_embed_dim, self.t_patch_size * 116 * in_chans)

        self.initialize_weights_trivial()
        ################################################
        # Decomp

        print("model initialized")

    def init_multiple_patch(self):

        self.Embedding_patch.multi_patch()

        self.head_layer_1 = nn.Sequential(*[
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias=not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias=not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.t_patch_size * 1 ** 2 * self.in_chans,
                      bias=not self.args.no_qkv_bias)
        ])

        self.head_layer_2 = nn.Sequential(*[
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias=not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias=not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.t_patch_size * 2 ** 2 * self.in_chans,
                      bias=not self.args.no_qkv_bias)
        ])

        self.head_layer_4 = nn.Sequential(*[
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias=not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias=not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.t_patch_size * 4 ** 2 * self.in_chans,
                      bias=not self.args.no_qkv_bias)
        ])

        self.initialize_weights_trivial()

    def init_prompt(self):
        self.spec_mlp = nn.Sequential(*[
            nn.Linear(self.args.his_len + 2, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        ])
        self.prompt_spatial_patch_t_1 = SpatialPatchEmb(self.embed_dim, self.embed_dim, 1)
        self.prompt_spatial_patch_f_1 = SpatialPatchEmb(self.embed_dim, self.embed_dim, 1)
        self.prompt_spatial_patch_t_2 = SpatialPatchEmb(self.embed_dim, self.embed_dim, 2)
        self.prompt_spatial_patch_f_2 = SpatialPatchEmb(self.embed_dim, self.embed_dim, 2)
        self.prompt_spatial_patch_t_4 = SpatialPatchEmb(self.embed_dim, self.embed_dim, 4)
        self.prompt_spatial_patch_f_4 = SpatialPatchEmb(self.embed_dim, self.embed_dim, 4)
        self.temporal_patch = nn.Conv1d(in_channels=1, out_channels=self.embed_dim, kernel_size=self.args.t_patch_size,
                                        stride=self.args.t_patch_size)
        encdoer_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=2, dim_feedforward=self.embed_dim,
                                                   batch_first=True)
        self.temporal_attn_encoder = nn.TransformerEncoder(encoder_layer=encdoer_layer, num_layers=1)
        self.temporaltokenConv = nn.Conv1d(in_channels=1, out_channels=self.embed_dim, kernel_size=3, stride=1,
                                           padding=1, padding_mode='circular', bias=False)


        # self.gcn_t = GAT(self.embed_dim, self.embed_dim, self.embed_dim)       ##替换为GAT---todo---
        # self.gcn_f = GAT(self.embed_dim, self.embed_dim, self.embed_dim)

        self.gcn_t = DCRNNWrapper(self.embed_dim, self.embed_dim, self.embed_dim, k=2)
        self.gcn_f = DCRNNWrapper(self.embed_dim, self.embed_dim, self.embed_dim, k=2)


        # self.gcn_topo_t = GCN(self.embed_dim, self.embed_dim, self.embed_dim)
        # self.gcn_topo_f = GCN(self.embed_dim, self.embed_dim, self.embed_dim)
        self.gcn_topo_t = DCRNNWrapper(self.embed_dim, self.embed_dim, self.embed_dim, k=2)
        self.gcn_topo_f = DCRNNWrapper(self.embed_dim, self.embed_dim, self.embed_dim, k=2)


        self.spec_liner = nn.Linear(2 * (self.args.his_len // 2 + 1), self.embed_dim)

        self.enc_memory_t = Memory(num_memory=self.args.num_memory, memory_dim=self.embed_dim, args=self.args)
        self.enc_memory_f = Memory(num_memory=self.args.num_memory, memory_dim=self.embed_dim, args=self.args)

        self.prompt_spatial_patch_t_1.apply(self._init_weights)
        self.prompt_spatial_patch_f_1.apply(self._init_weights)
        self.prompt_spatial_patch_t_2.apply(self._init_weights)
        self.prompt_spatial_patch_f_2.apply(self._init_weights)
        self.temporal_patch.apply(self._init_weights)
        self.gcn_t.apply(self._init_weights)
        self.gcn_f.apply(self._init_weights)
        self.gcn_topo_t.apply(self._init_weights)
        self.gcn_topo_f.apply(self._init_weights)

        self.enc_memory_t.apply(self._init_weights)
        self.enc_memory_f.apply(self._init_weights)

    def get_weights_sincos(self, num_t_patch, num_patch_1, num_patch_2):

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed_spatial.shape[-1],
            grid_size1=num_patch_1,
            grid_size2=num_patch_2
        )

        pos_embed_spatial = nn.Parameter(
            torch.zeros(1, num_patch_1 * num_patch_2, self.embed_dim)
        )
        pos_embed_temporal = nn.Parameter(
            torch.zeros(1, num_t_patch, self.embed_dim)
        )

        pos_embed_spatial.data.copy_(torch.tensor(pos_embed, dtype=torch.float32).unsqueeze(0))

        pos_temporal_emb = get_1d_sincos_pos_embed_from_grid(pos_embed_temporal.shape[-1],
                                                             np.arange(num_t_patch, dtype=np.float32))

        pos_embed_temporal.data.copy_(torch.tensor(pos_temporal_emb, dtype=torch.float32).unsqueeze(0))

        pos_embed_spatial.requires_grad = False
        pos_embed_temporal.requires_grad = False

        return pos_embed_spatial, pos_embed_temporal, copy.deepcopy(pos_embed_spatial), copy.deepcopy(
            pos_embed_temporal)

    def initialize_weights_trivial(self):
        torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

        torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

        torch.nn.init.normal_(self.mask_token, std=0.02)

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

    def patchify(self, imgs, patch_size):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, _, T, H, W = imgs.shape
        p = patch_size
        u = self.args.t_patch_size
        # assert H % p == 0 and W % p == 0 and T % u == 0
        h = H // p
        w = W // p
        t = T // u
        x = imgs.reshape(shape=(N, 1, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p ** 2 * 1))
        # self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

    def pos_embed_enc(self, ids_keep, batch, input_size):

        pos_embed_spatial, pos_embed_temporal, _, _ = self.get_weights_sincos(input_size[0], input_size[1],
                                                                              input_size[2])

        pos_embed = pos_embed_spatial[:, :input_size[1] * input_size[2]].repeat(
            1, input_size[0], 1
        ) + torch.repeat_interleave(
            pos_embed_temporal[:, :input_size[0]],
            input_size[1] * input_size[2],
            dim=1,
        )
        pos_embed = pos_embed.to(ids_keep.device)

        pos_embed = pos_embed.expand(batch, -1, -1)

        pos_embed_sort = torch.gather(
            pos_embed,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
        )

        return pos_embed_sort

    def pos_embed_dec(self, ids_keep, batch, input_size):

        _, _, decoder_pos_embed_spatial, decoder_pos_embed_temporal = self.get_weights_sincos(input_size[0],
                                                                                              input_size[1],
                                                                                              input_size[2])

        decoder_pos_embed = decoder_pos_embed_spatial[:, :input_size[1] * input_size[2]].repeat(
            1, input_size[0], 1
        ) + torch.repeat_interleave(
            decoder_pos_embed_temporal[:, :input_size[0]],
            input_size[1] * input_size[2],
            dim=1,
        )

        decoder_pos_embed = decoder_pos_embed.to(ids_keep.device)

        decoder_pos_embed = decoder_pos_embed.expand(batch, -1, -1)

        return decoder_pos_embed

    def forward_encoder(self, x, x_mark, mask_ratio, mask_strategy, seed=None, data=None, mode='backward', prompt={},
                        patch_size=1, split_nodes=None):
        # embed patches
        N, _, T, H, W = x.shape

        origin_x = x.clone()

        edges = prompt['topo']

        #修改
        # x:时空数据，x_mark:时间辅助信息，x(51,1,24,32,32)---->(),TimeEmb()

        if 'Graph' not in data:

            # x:时空数据，x_mark:时间辅助信息，x(51,1,24,32,32)---->(),TimeEmb()
            x, TimeEmb = self.Embedding_patch(x, x_mark, edges, is_time=self.args.is_time_emb, patch_size=patch_size,
                                              hour_num=data)
        else:
            x, TimeEmb = self.Embedding_patch_graph(x, x_mark, edges, split_nodes, is_time=self.args.is_time_emb,
                                                    patch_size=patch_size, hour_num=data)



        _, L, C = x.shape  # x(51,512,256),TimeEmb(51,512,256)

        T = T // self.args.t_patch_size

        # assert mode in ['backward','forward']

        x, mask, ids_restore, ids_keep = causal_masking(x, mask_ratio, T=T, mask_strategy=mask_strategy)

        if 'Graph' not in data:
            input_size = (T, H // patch_size, W // patch_size)
        else:
            input_size = (T, len(split_nodes), 1)

        # 生成位置编码
        pos_embed_sort = self.pos_embed_enc(ids_keep, N, input_size)
        # assert x.shape == pos_embed_sort.shape

        # 加上位置编码
        x_attn = x + pos_embed_sort

        prompt_save = {}
        attn_bias = {}




        # # 提前时域和频域信息（51,256,256）
        # prompt_t = self.enc_memory_t(prompt['t'].reshape(-1, prompt['t'].shape[-1]))
        # prompt_t = prompt_t['out'].reshape(prompt['t'].shape)
        #
        # prompt_f = self.enc_memory_f(prompt['f'].reshape(-1, prompt['f'].shape[-1]))
        # prompt_f = prompt_f['out'].reshape(prompt['f'].shape)
        #
        # # 自适应图(51,256,256)
        # adp_t = F.softmax(F.relu(prompt_t @ prompt_t.transpose(1, 2)), dim=-1)  # N * (H*W) * (H*W)
        # adp_f = F.softmax(F.relu(prompt_f @ prompt_f.transpose(1, 2)), dim=-1)
        #
        # data_list_t = []
        # data_list_f = []
        # edge_att_t, edge_att_f = [], []
        #
        # # 循环N生成边？？
        # for i in range(adp_t.size(0)):
        #     edge_index_t = adp_t[i].nonzero().t().contiguous() + i * adp_t.shape[1]
        #     data_list_t.append(edge_index_t)
        #     edge_att_t.append(adp_t[i][adp_t[i] != 0])
        #
        #     edge_index_f = adp_f[i].nonzero().t().contiguous() + i * adp_f.shape[1]
        #     data_list_f.append(edge_index_f)
        #     edge_att_f.append(adp_f[i][adp_f[i] != 0])
        #
        # edge_t = torch.cat(data_list_t, dim=-1)
        # edge_att_t = torch.cat(edge_att_t, dim=0)
        # edge_f = torch.cat(data_list_f, dim=-1)
        # edge_att_f = torch.cat(edge_att_f, dim=0)
        #
        # prompt_t = self.gcn_t(prompt_t.reshape(-1, prompt_t.shape[-1]), edge_t, edge_att_t).reshape(N,
        #                                                                                             H * W // patch_size ** 2,
        #                                                                                             self.embed_dim)
        # prompt_f = self.gcn_f(prompt_f.reshape(-1, prompt_f.shape[-1]), edge_f, edge_att_f).reshape(N,
        #                                                                                             H * W // patch_size ** 2,
        #                                                                                             self.embed_dim)
        #
        # # 输出：[51, 768, 256]
        # prompt_t = prompt_t.unsqueeze(1).repeat(1, self.args.his_len // self.args.t_patch_size, 1, 1).reshape(N,
        #                                                                                                       x_attn.shape[
        #                                                                                                           1],
        #                                                                                                       self.embed_dim)
        #
        # prompt_f = prompt_f.unsqueeze(1).repeat(1, self.args.his_len // self.args.t_patch_size, 1, 1).reshape(N,
        #                                                                                                       x_attn.shape[
        #                                                                                                           1],
        #                                                                                                       self.embed_dim)
        #
        # # assert prompt_t.shape == prompt_f.shape == x_attn.shape
        #
        # prompt_save['t'] = prompt_t.clone()
        # prompt_save['f'] = prompt_f.clone()
        #
        # prompt_t = self.enc_memory_t(prompt['t'].reshape(-1, prompt['t'].shape[-1]))
        # prompt_t = prompt_t['out'].reshape(prompt['t'].shape)
        #
        # prompt_f = self.enc_memory_f(prompt['f'].reshape(-1, prompt['f'].shape[-1]))
        # prompt_f = prompt_f['out'].reshape(prompt['f'].shape)
        #
        # prompt_t = prompt_t.unsqueeze(1).repeat(1, self.args.his_len // self.args.t_patch_size, 1, 1).reshape(N,
        #                                                                                                       x_attn.shape[
        #                                                                                                           1],
        #                                                                                                       self.embed_dim)
        # prompt_f = prompt_f.unsqueeze(1).repeat(1, self.args.his_len // self.args.t_patch_size, 1, 1).reshape(N,
        #                                                                                                       x_attn.shape[
        #                                                                                                           1],
        #                                                                                                       self.embed_dim)
        #
        # # assert prompt_t.shape == prompt_f.shape == x_attn.shape
        #
        # prompt_save['node_t'] = prompt_t.clone()
        # prompt_save['node_f'] = prompt_f.clone()



        if self.args.is_prompt == 1 and 'graph' in self.args.prompt_content:

            # 提前时域和频域信息（51,256,256）
            prompt_t = self.enc_memory_t(prompt['t'].reshape(-1, prompt['t'].shape[-1]))
            prompt_t = prompt_t['out'].reshape(prompt['t'].shape)

            prompt_f = self.enc_memory_f(prompt['f'].reshape(-1, prompt['f'].shape[-1]))
            prompt_f = prompt_f['out'].reshape(prompt['f'].shape)

            # 自适应图(51,256,256)
            adp_t = F.softmax(F.relu(prompt_t @ prompt_t.transpose(1, 2)), dim=-1)  # N * (H*W) * (H*W)
            adp_f = F.softmax(F.relu(prompt_f @ prompt_f.transpose(1, 2)), dim=-1)

            data_list_t = []
            data_list_f = []
            edge_att_t, edge_att_f = [], []

            # 循环N生成边？？
            for i in range(adp_t.size(0)):
                edge_index_t = adp_t[i].nonzero().t().contiguous() + i * adp_t.shape[1]
                data_list_t.append(edge_index_t)
                edge_att_t.append(adp_t[i][adp_t[i] != 0])

                edge_index_f = adp_f[i].nonzero().t().contiguous() + i * adp_f.shape[1]
                data_list_f.append(edge_index_f)
                edge_att_f.append(adp_f[i][adp_f[i] != 0])

            edge_t = torch.cat(data_list_t, dim=-1)
            edge_att_t = torch.cat(edge_att_t, dim=0)
            edge_f = torch.cat(data_list_f, dim=-1)
            edge_att_f = torch.cat(edge_att_f, dim=0)

            if 'Graph' not in data:


                prompt_t = self.gcn_t(prompt_t.reshape(-1, prompt_t.shape[-1]), edge_t, edge_att_t).reshape(N,
                                                                                                            H * W // patch_size ** 2,
                                                                                                            self.embed_dim)
                prompt_f = self.gcn_f(prompt_f.reshape(-1, prompt_f.shape[-1]), edge_f, edge_att_f).reshape(N,
                                                                                                            H * W // patch_size ** 2,
                                                                                                            self.embed_dim)

            else:
                prompt_t = self.gcn_topo_t(prompt_t.reshape(-1, prompt_t.shape[-1]), edge_t, edge_att_t).reshape(N,
                                                                                                                 len(split_nodes),
                                                                                                                 self.embed_dim)
                prompt_f = self.gcn_topo_f(prompt_f.reshape(-1, prompt_f.shape[-1]), edge_f, edge_att_f).reshape(N,
                                                                                                                 len(split_nodes),
                                                                                                                 self.embed_dim)



            #输出：[51, 768, 256]
            prompt_t = prompt_t.unsqueeze(1).repeat(1, self.args.his_len // self.args.t_patch_size, 1, 1).reshape(N,x_attn.shape[1],self.embed_dim)


            prompt_f = prompt_f.unsqueeze(1).repeat(1, self.args.his_len // self.args.t_patch_size, 1, 1).reshape(N,x_attn.shape[1],self.embed_dim)


            # assert prompt_t.shape == prompt_f.shape == x_attn.shape

            prompt_save['t'] = prompt_t.clone()
            prompt_save['f'] = prompt_f.clone()

        if self.args.is_prompt == 1 and 'node' in self.args.prompt_content:
            prompt_t = self.enc_memory_t(prompt['t'].reshape(-1, prompt['t'].shape[-1]))
            prompt_t = prompt_t['out'].reshape(prompt['t'].shape)

            prompt_f = self.enc_memory_f(prompt['f'].reshape(-1, prompt['f'].shape[-1]))
            prompt_f = prompt_f['out'].reshape(prompt['f'].shape)

            prompt_t = prompt_t.unsqueeze(1).repeat(1, self.args.his_len // self.args.t_patch_size, 1, 1).reshape(N,
                                                                                                                  x_attn.shape[
                                                                                                                      1],
                                                                                                                  self.embed_dim)
            prompt_f = prompt_f.unsqueeze(1).repeat(1, self.args.his_len // self.args.t_patch_size, 1, 1).reshape(N,
                                                                                                                  x_attn.shape[
                                                                                                                      1],
                                                                                                                  self.embed_dim)

            # assert prompt_t.shape == prompt_f.shape == x_attn.shape

            prompt_save['node_t'] = prompt_t.clone()
            prompt_save['node_f'] = prompt_f.clone()


        # trend_all = torch.zeros_like(x_attn)


            #[9, 315, 256]
          ##输入：torch.Size([51, 200, 256])
        for index, blk in enumerate(self.enblocks):
            # x_attn, trend_part=blk(x_attn, attn_bias=attn_bias)
            # trend_all=trend_all+trend_part
            x_attn = blk(x_attn, attn_bias=attn_bias)

        return x_attn, mask, ids_restore, input_size, TimeEmb, prompt_save

    def forward_decoder(self, x, x_mark, mask, ids_restore, mask_strategy, TimeEmb, input_size=None, data=None,
                        prompt_graph={}):
        N = x.shape[0]
        T, H, W = input_size

        # embed tokens
        x = self.decoder_embed(x)

        C = x.shape[-1]

        x = causal_restore(x, ids_restore, N, T, H, W, C, self.mask_token)

        decoder_pos_embed = self.pos_embed_dec(ids_restore, N, input_size)

        # add pos embed
        # assert x.shape == decoder_pos_embed.shape == TimeEmb.shape


        if self.args.is_time_emb == 1:
            x_attn = x + decoder_pos_embed + TimeEmb
        else:
            x_attn = x + decoder_pos_embed

        attn_bias = prompt_graph


          #修改
         # torch.Size([51, 768, 256]) torch.Size([51, 768, 256])
        prompt_t, prompt_f = prompt_graph['t'], prompt_graph['f']

        prompt_t = prompt_t.reshape(N, -1, H * W, prompt_t.shape[-1])[:, :1].repeat(1, (
                self.args.his_len + self.args.pred_len) // self.args.t_patch_size, 1, 1).reshape(N, -1,
                                                                                                 prompt_t.shape[
                                                                                                     -1])
        prompt_f = prompt_f.reshape(N, -1, H * W, prompt_f.shape[-1])[:, :1].repeat(1, (
                self.args.his_len + self.args.pred_len) // self.args.t_patch_size, 1, 1).reshape(N, -1,
                                                                                                 prompt_f.shape[
                                                                                                     -1])

        # if self.args.is_prompt == 1 and 'graph' in self.args.prompt_content:
        #
        #    #torch.Size([51, 768, 256]) torch.Size([51, 768, 256])
        #     prompt_t, prompt_f = prompt_graph['t'], prompt_graph['f']
        #
        #
        #     prompt_t = prompt_t.reshape(N, -1, H * W, prompt_t.shape[-1])[:, :1].repeat(1, (
        #                 self.args.his_len + self.args.pred_len) // self.args.t_patch_size, 1, 1).reshape(N, -1,
        #                                                                                                  prompt_t.shape[
        #                                                                                                      -1])
        #     prompt_f = prompt_f.reshape(N, -1, H * W, prompt_f.shape[-1])[:, :1].repeat(1, (
        #                 self.args.his_len + self.args.pred_len) // self.args.t_patch_size, 1, 1).reshape(N, -1,
        #                                                                                                  prompt_f.shape[
        #                                                                                                      -1])


            # assert x_attn.shape == prompt_t.shape == prompt_f.shape

            #修改
        x_attn += prompt_f + prompt_t
            # if 'graph_t' in self.args.prompt_content:
            #     x_attn = x_attn + prompt_t
            # elif 'graph_f' in self.args.prompt_content:
            #     x_attn = x_attn + prompt_f
            # else:
            #
            #     x_attn += prompt_f + prompt_t


        #修改
        prompt_t, prompt_f = prompt_graph['node_t'], prompt_graph['node_f']
        prompt_t = prompt_t.reshape(N, -1, H * W, prompt_t.shape[-1])[:, :1].repeat(1, (
                self.args.his_len + self.args.pred_len) // self.args.t_patch_size, 1, 1).reshape(N, -1,
                                                                                                 prompt_t.shape[
                                                                                                     -1])
        prompt_f = prompt_f.reshape(N, -1, H * W, prompt_f.shape[-1])[:, :1].repeat(1, (
                self.args.his_len + self.args.pred_len) // self.args.t_patch_size, 1, 1).reshape(N, -1,
                                                                                                 prompt_f.shape[
                                                                                                     -1])

        x_attn += prompt_f + prompt_t
        # if self.args.is_prompt == 1 and 'node' in self.args.prompt_content:
        #     prompt_t, prompt_f = prompt_graph['node_t'], prompt_graph['node_f']
        #     prompt_t = prompt_t.reshape(N, -1, H * W, prompt_t.shape[-1])[:, :1].repeat(1, (
        #                 self.args.his_len + self.args.pred_len) // self.args.t_patch_size, 1, 1).reshape(N, -1,
        #                                                                                                  prompt_t.shape[
        #                                                                                                      -1])
        #     prompt_f = prompt_f.reshape(N, -1, H * W, prompt_f.shape[-1])[:, :1].repeat(1, (
        #                 self.args.his_len + self.args.pred_len) // self.args.t_patch_size, 1, 1).reshape(N, -1,
        #                                                                                                  prompt_f.shape[
        #                                                                                                      -1])
        #
        #     x_attn += prompt_f + prompt_t


            # # assert x_attn.shape == prompt_t.shape == prompt_f.shape
            # if 'node_t' in self.args.prompt_content:
            #     x_attn = x_attn + prompt_t
            #
            # elif 'node_f' in self.args.prompt_content:
            #     x_attn = x_attn + prompt_f
            # else:
            #     x_attn += prompt_f + prompt_t

        # apply Transformer blocks
        trend_all = torch.zeros_like(x_attn)


        for index, blk in enumerate(self.deblocks):
            x_attn, trend_part = blk(x_attn, attn_bias=attn_bias)
            trend_all=trend_part+trend_all

        x_attn = self.decoder_norm(x_attn+trend_all)


        return x_attn

    def forward_loss(self, imgs, pred, mask, patch_size):
        """
        imgs: [N, 1, T, H, W]
        pred: [N, t*h*w, u*p*p*1]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """

        target = self.patchify(imgs, patch_size)

        # assert pred.shape == target.shape

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        loss1 = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss2 = (loss * (1 - mask)).sum() / (1 - mask).sum()
        return loss1, loss2, target

    def graph_loss(self, pred, target):
        # assert pred.shape == target.shape
        # assert pred.shape[1] == self.args.his_len + self.args.pred_len

        loss1 = ((pred[:, self.args.his_len:] - target[:, self.args.his_len:]) ** 2).mean()

        loss2 = ((pred[:, :self.args.his_len] - target[:, :self.args.his_len]) ** 2).mean()

        mask = torch.ones_like(target)

        mask[:, :self.args.his_len] = 0

        return loss1, loss2, target, mask

    def adpative_graph(self, img, img_mark, DataEmbedding, data, node_split=None, patch_size=2):
        N, _, T, H, W = img.shape
        # img_mark : N * T * 2

        img_origin = img.clone().squeeze(dim=1).reshape(N, T, H * W)  # (51,24,1024)
        img_origin = img_origin.permute(0, 2, 1).reshape(N * H * W, T)  # (N*H*W) * T    (52224,24)
        img_origin = img_origin[:, :self.args.his_len]   # only use history data      (52224,12)

        img_spec = torch.fft.rfft(img_origin, n=img_origin.shape[-1], norm="ortho",
                                  dim=-1)  # [N, K] K = T//2 + 1     (52224,7)
        img_spec = img_spec.reshape(N, H, W, self.args.his_len // 2 + 1)  # (51,32,32,7)

        img_spec = torch.cat((img_spec.real, img_spec.imag), dim=-1)  # [N, H, W, 2(his_len//2+1)]   (51,32,32,14)

        img_spec = self.spec_liner(img_spec)  # (51,32,32,256)

        img_tmp = img_origin.unsqueeze(1)  # (52224,1,12)

        img_tmp = self.temporaltokenConv(img_tmp).permute(0, 2, 1)  # N * T * Embed    (52224,12,256)

        img_mark = img_mark[:, :self.args.his_len].unsqueeze(dim=1).repeat(1, H * W, 1, 1).reshape(N * H * W,
                                                                                                   self.args.his_len,
                                                                                                   2)  # (52224,12,2)

        temporal_emb = DataEmbedding.temporal_emb(img_mark, data)  # 52224,12,256

        # assert img_tmp.shape == temporal_emb.shape
        img_tmp += temporal_emb  # 52224,12,256

        #
        img_tmp = torch.cat([self.temporal_attn_encoder(img_tmp[index:index + H * W]) for index in range(0, img_tmp.shape[0], H * W)],axis=0)[:, 0]  # 52224,256


        img_tmp = img_tmp.reshape(N, H, W, img_tmp.shape[-1])  # 51,32,32,256

        # #修改
        # img_spec = self.prompt_spatial_patch_f_2(img_spec.permute(0, 3, 1, 2))  # 51,256,256
        # img_tmp = self.prompt_spatial_patch_t_2(img_tmp.permute(0, 3, 1, 2))  # 51,256,256
        if 'Graph' not in data:
            if patch_size == 1:
                img_spec = self.prompt_spatial_patch_f_1(img_spec.permute(0, 3, 1, 2))
                img_tmp = self.prompt_spatial_patch_t_1(img_tmp.permute(0, 3, 1, 2))
            elif patch_size == 2:
                img_spec = self.prompt_spatial_patch_f_2(img_spec.permute(0, 3, 1, 2))  # 51,256,256
                img_tmp = self.prompt_spatial_patch_t_2(img_tmp.permute(0, 3, 1, 2))  # 51,256,256
            elif patch_size == 4:
                img_spec = self.prompt_spatial_patch_f_4(img_spec.permute(0, 3, 1, 2))
                img_tmp = self.prompt_spatial_patch_t_4(img_tmp.permute(0, 3, 1, 2))
        else:
            # patchify
            max_len = max([len(i) for i in node_split])   #修改
            n_group = len(node_split)

            img_tmp = torch.cat([torch.mean(torch.gather(img_tmp, 1,
                                                         group.view(1, group.shape[0], 1, 1).expand(img_tmp.shape[0],
                                                                                                    group.shape[0],
                                                                                                    img_tmp.shape[2],
                                                                                                    img_tmp.shape[
                                                                                                        3]).to(
                                                             img_tmp).long()), dim=1, keepdim=True) for group in
                                 node_split], dim=1).squeeze(dim=2)

            img_spec = torch.cat([torch.mean(torch.gather(img_spec, 1,
                                                          group.view(1, group.shape[0], 1, 1).expand(img_spec.shape[0],
                                                                                                     group.shape[0],
                                                                                                     img_spec.shape[2],
                                                                                                     img_spec.shape[
                                                                                                         3]).to(
                                                              img_tmp).long()), dim=1, keepdim=True) for group in
                                  node_split], dim=1).squeeze(dim=2)

        return img_tmp, img_spec

    # Create target key padding mask
    def create_padding_mask(self, seq_lengths, max_len):
        padding_mask = torch.zeros((len(seq_lengths), max_len), dtype=torch.bool)
        for i, length in enumerate(seq_lengths):
            padding_mask[i, length:] = True
        return padding_mask

    def forward(self, imgs, mask_ratio=0.5, mask_strategy='causal', seed=520, data='none', mode='backward', topo=None,
                subgraphs=None, patch_size=100):
        '''
        backward: 没有特定evaluation约束，forward: 有特定evaluation约束
        '''
        imgs, imgs_mark = imgs



        # img_tmp, img_spec = self.adpative_graph(imgs, imgs_mark, self.Embedding_patch, data=data,
        #                                                 patch_size=patch_size)  # 时域/频域

        if self.args.is_prompt == 1:
            if 'Graph' not in data:
                img_tmp, img_spec = self.adpative_graph(imgs, imgs_mark, self.Embedding_patch, data=data,
                                                        patch_size=patch_size)  # 时域/频域
            else:

                img_tmp, img_spec = self.adpative_graph(imgs, imgs_mark, self.Embedding_patch_graph, data=data,
                                                        node_split=subgraphs, patch_size=patch_size)
        else:
            img_tmp = None
            img_spec = None



        T, H, W = imgs.shape[2:]



        latent, mask, ids_restore, input_size, TimeEmb, prompt = self.forward_encoder(imgs, imgs_mark, mask_ratio,
                                                                                      mask_strategy, seed=seed,
                                                                                      data=data, mode=mode,
                                                                                      prompt={'t': img_tmp,
                                                                                              'f': img_spec,
                                                                                              'topo': topo},
                                                                                      patch_size=patch_size,
                                                                                      split_nodes=subgraphs)

        pred = self.forward_decoder(latent, imgs_mark, mask, ids_restore, mask_strategy, TimeEmb, input_size=input_size,
                                    data=data, prompt_graph=prompt)  # [N, L, p*p*1]

        L = pred.shape[1]



        if 'Graph' not in data:
            if patch_size == 1:
                pred = self.head_layer_1(pred)
            elif patch_size == 2:
                pred = self.head_layer_2(pred)
            elif patch_size == 4:
                pred = self.head_layer_4(pred)

        else:
            seq_lengths = [len(i) for i in subgraphs]
            max_len = max(seq_lengths)

            # if 'GraphBJ' in data:
            #     pred = self.pred_model_linear_GraphBJ(pred).reshape(pred.shape[0], T // self.args.t_patch_size,
            #                                                         len(subgraphs), self.args.t_patch_size, -1).permute(
            #         0, 1, 3, 2, 4)
            if 'GraphSH' in data:
                pred = self.pred_model_linear_GraphSH(pred).reshape(pred.shape[0], T // self.args.t_patch_size,
                                                                    len(subgraphs), self.args.t_patch_size, -1).permute(
                    0, 1, 3, 2, 4)
            # elif 'GraphNJ' in data:
            #     pred = self.pred_model_linear_GraphNJ(pred).reshape(pred.shape[0], T // self.args.t_patch_size,
            #                                                         len(subgraphs), self.args.t_patch_size, -1).permute(
            #         0, 1, 3, 2, 4)
            elif 'GraphPEMS' in data:
                pred = self.pred_model_linear_GraphPEMS(pred).reshape(pred.shape[0], T // self.args.t_patch_size,
                                                                    len(subgraphs), self.args.t_patch_size, -1).permute(
                    0, 1, 3, 2, 4)


            pred = pred.reshape(pred.shape[0], T, len(subgraphs), -1)

            pred = torch.cat([pred[:, :, g, :seq_lengths[g]] for g in range(pred.shape[2])], dim=2)

            target = imgs.squeeze(dim=(1, 4))

            target = torch.cat([torch.gather(target, 2,
                                             group.view(1, 1, group.shape[0]).expand(target.shape[0], target.shape[1],
                                                                                     group.shape[0]).to(target).long())
                                for group in subgraphs], dim=2)

        if 'Graph' not in data:
            loss1, loss2, target = self.forward_loss(imgs, pred, mask, patch_size)

        else:
            loss1, loss2, target, mask = self.graph_loss(pred, target)

        return loss1, loss2, pred, target, mask

