import time
from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import numpy as np
from openpyxl.descriptors.serialisable import seq_types
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import DropPath, Mlp, Attention

from Embed import GraphEmbedding, DataEmbedding, TokenEmbedding, SpatialPatchEmb, get_2d_sincos_pos_embed, \
    get_2d_sincos_pos_embed_with_resolution, get_1d_sincos_pos_embed_from_grid, \
    get_1d_sincos_pos_embed_from_grid_with_resolution
from mask_strategy import *
import copy
from Autoformer_EncDec import  series_decomp
from Prompt_network import DCRNNWrapper,MultiScaleMemory


# def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
#     """
#     get modes on frequency domain:
#     'random' means sampling randomly;
#     'else' means sampling the lowest modes;
#     """
#     modes = min(modes, seq_len//2)
#     if mode_select_method == 'random':
#         index = list(range(0, seq_len // 2))
#         np.random.shuffle(index)
#         index = index[:modes]
#     else:
#         index = list(range(0, modes))
#     index.sort()
#     return index

       ###--todo 修改2
def get_frequency_modes(modes=64,mode_select_method=1, x_ft=None):
    """
    专为时空预测设计的代表性频率采样：
    Trend + Seasonal + Residual（SR）三路采样。
    """

    if mode_select_method == 2:
        Lf = x_ft.shape[-1]

        # ===== 1. FFT 和能量/残差计算 =====
        amp = x_ft.abs()
        # energy 包含 DC（索引0），但下面为方便起见我们保留 DC（可被采样）
        energy = (amp ** 2).mean(dim=(0, 1,2))  # shape (Lf+1,)
        # 如果 energy 长度不对，修正 Lf

        energy = energy[:Lf]

        # SR 计算（spectral residual），与之前逻辑一致
        log_amp = torch.log1p(amp.mean(dim=(0, 1,2)))[:Lf]
        smooth = torch.nn.functional.avg_pool1d(
            log_amp.view(1, 1, -1), kernel_size=9, stride=1, padding=4
        ).view(-1)
        SR = torch.abs(log_amp - smooth)
        SR_l = SR.shape[0]

        # ===== 2. 频段定义（可调整比例） =====
        trend_l, trend_r = 0, max(1, int(0.1 * Lf))
        residual_l, residual_r = int(0.6 * SR_l), SR_l

        # ===== 3. 分配数量（初步分配，随后会 clamp） =====
        n_trend = int(0.3 * modes)
        n_season = int(0.3 * modes)
        n_residual = modes - n_trend - n_season

        # safety: clamp to available counts
        def clamp_k(k, available):
            return max(0, min(k, available))

        def safe_multinomial_from_tensor(score_tensor, k):
            """
            score_tensor: 1D tensor (n,)
            返回 numpy list 的索引（相对于这个段的偏移为0）
            """

            n = score_tensor.size(0)

            if n == 0 or k <= 0:
                print('报错!!!!')
                return []
            k = min(k, n)
            prob = torch.softmax(score_tensor, dim=0)
            idx = torch.multinomial(prob, k, replacement=False)
            return idx.cpu().numpy().tolist()

        selected = []

        # ===== (1) Trend 低频 =====
        if trend_r - trend_l > 0:
            avail = trend_r - trend_l
            k = clamp_k(n_trend, avail)
            idx = safe_multinomial_from_tensor(energy[trend_l:trend_r], k)
            # 转为全局索引
            selected.extend([i + trend_l for i in idx])

        # ===== (2) Seasonal：基于周期先验 =====
        seasonal_freqs = []

        def add_freq(period):
            # 计算期望频率索引（向下取整）
            f = int(Lf / period)
            if 0 <= f < Lf:
                seasonal_freqs.append(f)

        # 常见周期：根据你的时空任务调整此列表
        for p in [2, 5, 7, 12, 24, 36, 48, 96, 144, 288]:
            add_freq(p)

        # # 去重并限制数量
        # seasonal_freqs = list(dict.fromkeys(seasonal_freqs))  # 保持顺序去重

        # if len(seasonal_freqs) > n_season:
        #     seasonal_freqs = seasonal_freqs[:n_season]
        # selected.extend(seasonal_freqs)

        # ===== (3) Residual 高频（用 SR） =====

        if residual_r - residual_l > 0:
            avail = residual_r - residual_l
            k = clamp_k(n_residual, avail)

            idx = safe_multinomial_from_tensor(SR[residual_l:residual_r], k)
            selected.extend([i + residual_l for i in idx])

        # # ===== 5. 去重 + 如果不足 modes，从剩余频率随机补足 =====
        # selected = list(dict.fromkeys(selected))  # 去重并保持顺序
        # # 过滤掉超界的（保险）
        # selected = [int(i) for i in selected if 0 <= i < Lf]
        # print('Residual', len(selected))

        # 如果不足，需要补足（从剩余索引中随机抽取）

        remaining = [i for i in range(0, Lf) if i not in selected]
        need = modes - len(selected)
        if len(remaining) > 0 and need > 0:
            # 若remaining长度小于need，则取全部
            take = min(need, len(remaining))
            # 随机补齐（与原逻辑保持一致：随机）
            add = list(np.random.choice(remaining, size=take, replace=False))
            selected.extend(add)
        # if len(selected) < modes:
        #
        #     remaining = [i for i in range(0, Lf) if i not in selected]
        #     need = modes - len(selected)
        #     if len(remaining) > 0 and need > 0:
        #         # 若remaining长度小于need，则取全部
        #         take = min(need, len(remaining))
        #         # 随机补齐（与原逻辑保持一致：随机）
        #         add = list(np.random.choice(remaining, size=take, replace=False))
        #         selected.extend(add)

        # 最终截取到 modes 个（若多于 modes，截断）
        # if len(selected) > modes:
        #     selected = selected[:modes]

        # 保证返回类型与原函数一致：排序的 python list
        selected = sorted(int(i) for i in selected)
        return selected


        # seq_len = x.size(-1)
        #
        # Lf = seq_len//2+1
        # modes = min(modes, Lf)
        #
        # x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1, x.size(3))
        #
        # # ===== 1. FFT 和能量/残差计算 =====
        # x_ft = torch.fft.rfft(x, dim=1)  # (B, Lf+1, C)
        #
        # amp = x_ft.abs()
        # # energy 包含 DC（索引0），但下面为方便起见我们保留 DC（可被采样）
        # energy = (amp ** 2).mean(dim=(0, 2))  # shape (Lf+1,)
        # # 如果 energy 长度不对，修正 Lf
        #
        # energy = energy[:Lf]
        #
        # # SR 计算（spectral residual），与之前逻辑一致
        # log_amp = torch.log1p(amp.mean(dim=(0, 2)))[:Lf]
        # smooth = torch.nn.functional.avg_pool1d(
        #     log_amp.view(1, 1, -1), kernel_size=9, stride=1, padding=4
        # ).view(-1)
        # SR = torch.abs(log_amp - smooth)
        #
        # SR_l=SR.shape[0]
        #
        # # ===== 2. 频段定义（可调整比例） =====
        # trend_l, trend_r = 0, max(1, int(0.1 * Lf))
        # residual_l, residual_r = int(0.6 * SR_l), SR_l
        #
        # # ===== 3. 分配数量（初步分配，随后会 clamp） =====
        # n_trend = int(0.3 * modes)
        # n_season = int(0.3 * modes)
        # n_residual = modes - n_trend - n_season
        #
        # # safety: clamp to available counts
        # def clamp_k(k, available):
        #     return max(0, min(k, available))
        #
        # # # ===== 4. 采样函数（保证不越界） =====
        # # def safe_multinomial_from_tensor(score_tensor, k):
        # #     """
        # #     score_tensor: 1D tensor (n,)
        # #     返回 python list
        # #     """
        # #     n = score_tensor.size(0)
        # #     if n == 0 or k <= 0:
        # #         return []
        # #
        # #     k = min(k, n)
        # #
        # #     # ===== 关键：数值安全 =====
        # #     score_tensor = score_tensor.clone()
        # #
        # #     # 1. 替换 NaN / Inf
        # #     score_tensor[~torch.isfinite(score_tensor)] = -1e9
        # #
        # #     # 2. 若全相等或全极小，改用 uniform
        # #     if torch.all(score_tensor == score_tensor[0]):
        # #         print(score_tensor)
        # #         prob = torch.ones_like(score_tensor) / n
        # #     else:
        # #         prob = torch.softmax(score_tensor, dim=0)
        # #
        # #     # 3. 再保险一次
        # #     if not torch.isfinite(prob).all() or prob.sum() <= 0:
        # #         prob = torch.ones_like(score_tensor) / n
        # #
        # #     idx = torch.multinomial(prob, k, replacement=False)
        # #     return idx.cpu().tolist()
        #
        # def safe_multinomial_from_tensor(score_tensor, k):
        #     """
        #     score_tensor: 1D tensor (n,)
        #     返回 numpy list 的索引（相对于这个段的偏移为0）
        #     """
        #
        #     n = score_tensor.size(0)
        #
        #     if n == 0 or k <= 0:
        #         print('报错!!!!')
        #         return []
        #     k = min(k, n)
        #     prob = torch.softmax(score_tensor, dim=0)
        #     idx = torch.multinomial(prob, k, replacement=False)
        #     return idx.cpu().numpy().tolist()
        #
        # selected = []
        #
        # # ===== (1) Trend 低频 =====
        # if trend_r - trend_l > 0:
        #     avail = trend_r - trend_l
        #     k = clamp_k(n_trend, avail)
        #     idx = safe_multinomial_from_tensor(energy[trend_l:trend_r], k)
        #     # 转为全局索引
        #     selected.extend([i + trend_l for i in idx])
        #
        # # ===== (2) Seasonal：基于周期先验 =====
        # seasonal_freqs = []
        #
        # def add_freq(period):
        #     # 计算期望频率索引（向下取整）
        #     f = int(seq_len / period)
        #     if 0 <= f < Lf:
        #         seasonal_freqs.append(f)
        #
        #
        # # 常见周期：根据你的时空任务调整此列表
        # for p in [2,5,7,12,24,36,48,96,144,288]:
        #     add_freq(p)
        #
        # # # 去重并限制数量
        # # seasonal_freqs = list(dict.fromkeys(seasonal_freqs))  # 保持顺序去重
        #
        # # if len(seasonal_freqs) > n_season:
        # #     seasonal_freqs = seasonal_freqs[:n_season]
        # # selected.extend(seasonal_freqs)
        #
        # # ===== (3) Residual 高频（用 SR） =====
        #
        #
        # if residual_r - residual_l > 0:
        #     avail = residual_r - residual_l
        #     k = clamp_k(n_residual, avail)
        #
        #     idx = safe_multinomial_from_tensor(SR[residual_l:residual_r], k)
        #     selected.extend([i + residual_l for i in idx])
        #
        #
        # # # ===== 5. 去重 + 如果不足 modes，从剩余频率随机补足 =====
        # # selected = list(dict.fromkeys(selected))  # 去重并保持顺序
        # # # 过滤掉超界的（保险）
        # # selected = [int(i) for i in selected if 0 <= i < Lf]
        # # print('Residual', len(selected))
        #
        # # 如果不足，需要补足（从剩余索引中随机抽取）
        #
        # remaining = [i for i in range(0, Lf) if i not in selected]
        # need = modes - len(selected)
        # if len(remaining) > 0 and need > 0:
        #     # 若remaining长度小于need，则取全部
        #     take = min(need, len(remaining))
        #     # 随机补齐（与原逻辑保持一致：随机）
        #     add = list(np.random.choice(remaining, size=take, replace=False))
        #     selected.extend(add)
        # # if len(selected) < modes:
        # #
        # #     remaining = [i for i in range(0, Lf) if i not in selected]
        # #     need = modes - len(selected)
        # #     if len(remaining) > 0 and need > 0:
        # #         # 若remaining长度小于need，则取全部
        # #         take = min(need, len(remaining))
        # #         # 随机补齐（与原逻辑保持一致：随机）
        # #         add = list(np.random.choice(remaining, size=take, replace=False))
        # #         selected.extend(add)
        #
        # # 最终截取到 modes 个（若多于 modes，截断）
        # # if len(selected) > modes:
        # #     selected = selected[:modes]
        #
        # # 保证返回类型与原函数一致：排序的 python list
        # selected = sorted(int(i) for i in selected)
        #
        # return selected
    # ============= random sampling =============
    # modes = min(modes, Lf)
    if mode_select_method == 1:
        Lf=200
        index = list(range(0, Lf))
        np.random.shuffle(index)
        index = index[:modes]

    # ============= lowest freqs =============
    else:
        print("报错!")
        index = list(range(0, modes))
        exit()

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
    print("正在使用UniFlow_A_D_M_F_I模型！！！")
    if args.size == 'small':
        model = UniFlow_A_D_M_F_I(
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
        model = UniFlow_A_D_M_F_I(
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
        model = UniFlow_A_D_M_F_I(
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


#########################################################
# FourierBlock（支持 importance）
#########################################################
class FourierBlock(nn.Module):
    def __init__(self, num_heads, in_channels, out_channels, seq_len,
                 modes=0, mode_select_method=1, d_keys=None, d_values=None):
        super(FourierBlock, self).__init__()
        d_keys = d_keys or (in_channels // num_heads)
        d_values = d_values or (in_channels // num_heads)

        print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        self.modes = modes
        self.mode_select_method = mode_select_method
        self.seq_len = seq_len

        # 初始 index（不用 importance）
        self.index = get_frequency_modes(modes, mode_select_method=1)
        print('initialize modes={}, index={}'.format(modes, self.index))

        self.scale = (1 / (in_channels * out_channels))
        self.num_heads = num_heads

        # 用 modes 大小初始化 weights1（固定参数，不随 index 动态变化）
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                num_heads,
                in_channels // num_heads,
                out_channels // num_heads,
                modes,
                dtype=torch.cfloat
            )
        )

        # linear projection
        self.query_projection = nn.Linear(in_channels, d_keys * num_heads)
        self.key_projection = nn.Linear(in_channels, d_keys * num_heads)
        self.value_projection = nn.Linear(in_channels, d_values * num_heads)

    # complex multiplication
    def compl_mul1d(self, input, weights):
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, x, mask=None):
        queries = x
        B, L, _ = queries.shape
        H = self.num_heads

        # projection
        queries = self.query_projection(queries).view(B, L, H, -1)   # B, L, H, E
        _, _, H, E = queries.shape

        # permute → Fourier 变换维度为 L

        x = queries.permute(0, 2, 3, 1)   # B, H, E, L




        # rfft
        x_ft = torch.fft.rfft(x, dim=-1)   # B, H, E, Lf+1


        self.index = get_frequency_modes(
            modes=self.modes,
            mode_select_method=self.mode_select_method,
            x_ft=x_ft,  #传入FFT所需序列
        )


        # 创建输出频域张量
        out_ft = torch.zeros(B, H, E, self.modes, device=x.device, dtype=torch.cfloat)

        # 逐频率卷积
        for wi, fi in enumerate(self.index):
            out_ft[..., wi] = self.compl_mul1d(x_ft[..., fi], self.weights1[..., wi])

        # ifft
        out_ft = torch.fft.irfft(out_ft, n=L)

        # reshape 回 batch format
        out = out_ft.view(B, L, -1)
        return out


class FourierCrossAttention(nn.Module):
    def __init__(self, num_heads, in_channels, out_channels,
                 seq_len_q, seq_len_kv,
                 modes=64, mode_select_method=1,
                 activation='tanh', policy=0,
                 d_keys=None, d_values=None):

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
        self.num_heads = num_heads

        # 保存参数（forward 里会动态更新 index）
        self.modes = modes
        self.mode_select_method = mode_select_method
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv

        # 初始化 index（仅用于检查维度）
        self.index_q = get_frequency_modes(modes=modes, mode_select_method=1)
        self.index_kv = get_frequency_modes(modes=modes, mode_select_method=1)

        print(f'Init modes_q={len(self.index_q)}, index_q={self.index_q}')
        print(f'Init modes_kv={len(self.index_kv)}, index_kv={self.index_kv}')

        self.scale = 1.0 / (in_channels * out_channels)

        # 频率卷积参数（固定用 modes 长度）
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                num_heads,
                in_channels // num_heads,
                out_channels // num_heads,
                modes,
                dtype=torch.cfloat
            )
        )

        # projections
        self.query_projection = nn.Linear(in_channels, d_keys * num_heads)
        self.key_projection = nn.Linear(in_channels, d_keys * num_heads)
        self.value_projection = nn.Linear(in_channels, d_values * num_heads)

    # complex multiplication
    def compl_mul1d(self, input, weights):
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, x, mask=None):

        #############################################################
        # 1) Regular attention projections
        #############################################################
        queries = x
        keys = x
        values = x

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.num_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        B, L, H, E = queries.shape

        # For FFT permute -> (B, H, E, L/S)
        xq = queries.permute(0, 2, 3, 1)
        xk = keys.permute(0, 2, 3, 1)
        xv = values.permute(0, 2, 3, 1)

        #############################################################
        # 2) 动态更新频率 index（支持 importance）
        #############################################################




        #############################################################
        # 3) FFT & 选择频率
        #############################################################
        xq_ft = torch.fft.rfft(xq, dim=-1)
        xk_ft = torch.fft.rfft(xk, dim=-1)


        self.index_q = get_frequency_modes(
            modes=self.modes,
            mode_select_method=self.mode_select_method,
            x_ft=xq_ft
        )

        self.index_kv = get_frequency_modes(
            modes=self.modes,
            mode_select_method=self.mode_select_method,
            x_ft=xk_ft
        )

        Lf_q = len(self.index_q)
        Lf_kv = len(self.index_kv)


        # 只取需要的频率
        xq_ft_ = xq_ft[..., self.index_q]        # (B, H, E, modes)
        xk_ft_ = xk_ft[..., self.index_kv]       # (B, H, E, modes)

        #############################################################
        # 4) Frequency-domain attention
        #############################################################
        # xq_ft_: (B,H,E,F)   xk_ft_: (B,H,E,F)
        # 注意力权重：einsum(bhef,bhey)->bhxy
        xqk_ft = torch.einsum("bhef,bhey->bhfy", xq_ft_, xk_ft_)

        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()

        elif self.activation == 'softmax':
            xqk_abs = torch.softmax(xqk_ft.abs(), dim=-1)
            xqk_ft = torch.complex(xqk_abs, torch.zeros_like(xqk_abs))

        else:
            raise Exception(f"{self.activation} activation not supported")

        # xqkv_ft = A * Values
        xqkv_ft = torch.einsum("bhfy,bhey->bhef", xqk_ft, xk_ft_)

        #############################################################
        # 5) Frequency-wise linear transform
        #############################################################
        # weights1: (H, E_in, E_out, modes)
        xqkvw = torch.einsum("bhef,heof->bhof", xqkv_ft, self.weights1)

        #############################################################
        # 6) Scatter 回完整频率维度 (L//2+1)
        #############################################################
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)

        for fi, fq in enumerate(self.index_q):
            out_ft[..., fq] = xqkvw[..., fi]

        #############################################################
        # 7) IFFT back to time domain
        #############################################################
        out = torch.fft.irfft(out_ft / (self.in_channels * self.out_channels), n=L)

        # reshape 回 (B, L, C)
        out = out.view(B, L, -1)
        return out


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
            modes=64,
            attn_drop=0.01,
            drop_path=0.05,  #0
            act_layer=nn.GELU,  # 激活函数，默认为 GELU
            norm_layer=nn.LayerNorm,
            # cross_attention=AutoCorrelationWrapper,  # 自定义的注意力机制类，默认为 Attention
            self_attention=FourierBlock

    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = self_attention(num_heads=num_heads,
                                 in_channels=256,
                                        out_channels=256,
                                        seq_len=12,
                                        modes=modes,
                                        mode_select_method=2)

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
            modes=64,
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

        self.seq_len=12
        self.pred_len=12

        self.attn = self_attention(num_heads=num_heads,
                                 in_channels=256,
                                        out_channels=256,
                                        seq_len=self.seq_len // 2 + self.pred_len,
                                        modes=modes,
                                        mode_select_method=2)
        self.correlation = cross_attention(num_heads=num_heads,
                                           in_channels=256,
                                                  out_channels=256,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=modes,
                                                  mode_select_method=2)



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
        x,trend3 = self.decomp3(x+y)

        residual_trend = trend1 + trend2 + trend3

        #residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)

        #seasonal_init, trend_init = self.decomp(x)   #趋势分解
        #输出[51, 768, 256]

        return x, residual_trend


class DifferentiableGraphPartition(nn.Module):
    """
    可微图划分模块 (替代 pymetis)，既能反向传播，又能输出subgraphs
    """

    def __init__(self, num_nodes, embed_dim=128, num_clusters=16, tau=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.tau = tau  # 温度控制聚类“硬度”

        # 节点嵌入
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.nodevec2 = nn.Parameter(torch.randn(embed_dim, num_nodes))

        # 聚类中心 (可学习)
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, embed_dim))

    def forward(self, node_features=None):
        """
        返回：
        - adj_dynamic: 可学习邻接矩阵 [N, N]
        - S_soft: 节点的soft cluster assignment [N, K]
        - subgraphs: 离散节点分组（列表），不参与梯度
        """
        # === Step 1: 可学习邻接矩阵 ===
        adj_dynamic = F.relu(torch.mm(self.nodevec1, self.nodevec2))
        adj_dynamic = F.softmax(adj_dynamic, dim=1)

        # === Step 2: 可微聚类分配 ===
        # 使用节点嵌入与聚类中心计算相似度
        if node_features is not None:
            node_embed = node_features
        else:
            node_embed = self.nodevec1

        sim = torch.matmul(node_embed, self.cluster_centers.T) / self.tau
        S_soft = F.softmax(sim, dim=-1)  # [N, K] 可微

        # === Step 3: 离散分配用于生成subgraphs（无梯度） ===
        hard_assign = torch.argmax(S_soft, dim=-1).detach().cpu().numpy()
        subgraphs = []
        for k in range(self.num_clusters):
            group = torch.tensor((hard_assign == k).nonzero()[0])
            if len(group) > 0:
                subgraphs.append(group)

        return adj_dynamic, S_soft, subgraphs



class UniFlow_A_D_M_F_I(nn.Module):
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


        self.enblocks = nn.ModuleList(  ##编码器
            [
                EncoderBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    modes=self.args.flag,
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
                    modes=self.args.flag,
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

        encdoer_layer2 = nn.TransformerEncoderLayer(d_model=decoder_embed_dim, nhead=2,
                                                    dim_feedforward=decoder_embed_dim // 2, batch_first=True)
        self.spatial_attn_spec_tmp = nn.TransformerEncoder(encoder_layer=encdoer_layer2, num_layers=1)

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.pred_model = TransformerDecoderModel(d_model=decoder_embed_dim, dim_feedforward=decoder_embed_dim // 2,
                                                  nhead=2, num_decoder_layers=1)
        self.pred_model_linear_GraphBJ = nn.Linear(decoder_embed_dim, self.t_patch_size * 105 * in_chans)
        self.pred_model_linear_GraphNJ = nn.Linear(decoder_embed_dim, self.t_patch_size * 105 * in_chans)
        self.pred_model_linear_GraphSH = nn.Linear(decoder_embed_dim, self.t_patch_size * 210 * in_chans)
        self.pred_model_linear_GraphPems = nn.Linear(decoder_embed_dim, self.t_patch_size * self.args.num_nodes * in_chans)

        self.graph_partition = DifferentiableGraphPartition(
            num_nodes=self.args.num_nodes,
            embed_dim=512,  # 512
            num_clusters=self.args.num_nodes // self.args.patch_size,
            tau=0.4
        )
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

        self.head_layer_3 = nn.Sequential(*[
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias=not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias=not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(self.decoder_embed_dim, self.t_patch_size * 3 ** 2 * self.in_chans,
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
        self.prompt_spatial_patch_t_3 = SpatialPatchEmb(self.embed_dim, self.embed_dim, 3)
        self.prompt_spatial_patch_f_3 = SpatialPatchEmb(self.embed_dim, self.embed_dim, 3)

        # self.prompt_spatial_patch_t_4 = SpatialPatchEmb(self.embed_dim, self.embed_dim, 4)
        # self.prompt_spatial_patch_f_4 = SpatialPatchEmb(self.embed_dim, self.embed_dim, 4)
        self.temporal_patch = nn.Conv1d(in_channels=1, out_channels=self.embed_dim, kernel_size=self.args.t_patch_size,
                                        stride=self.args.t_patch_size)
        encdoer_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=2, dim_feedforward=self.embed_dim,
                                                   batch_first=True)
        self.temporal_attn_encoder = nn.TransformerEncoder(encoder_layer=encdoer_layer, num_layers=1)
        self.temporaltokenConv = nn.Conv1d(in_channels=1, out_channels=self.embed_dim, kernel_size=3, stride=1,
                                           padding=1, padding_mode='circular', bias=False)


        self.gcn_t = DCRNNWrapper(self.embed_dim, self.embed_dim, self.embed_dim, k=2)
        self.gcn_f = DCRNNWrapper(self.embed_dim, self.embed_dim, self.embed_dim, k=2)
        self.gcn_topo_t = DCRNNWrapper(self.embed_dim, self.embed_dim, self.embed_dim, k=2)
        self.gcn_topo_f = DCRNNWrapper(self.embed_dim, self.embed_dim, self.embed_dim, k=2)


        self.spec_liner = nn.Linear(2 * (self.args.his_len // 2 + 1), self.embed_dim)

        # self.enc_memory_t = Memory(num_memory=self.args.num_memory, memory_dim=self.embed_dim, args=self.args)    #替换为多尺度版本----todo-@有效果
        # self.enc_memory_f = Memory(num_memory=self.args.num_memory, memory_dim=self.embed_dim, args=self.args)
        self.enc_memory_t = MultiScaleMemory(num_memory=self.args.num_memory, memory_dim=self.embed_dim,n_scales=3,args=self.args)
        self.enc_memory_f = MultiScaleMemory(num_memory=self.args.num_memory, memory_dim=self.embed_dim,n_scales=3, args=self.args)

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
        x: (N, L, patch_size**2 *3)            对输入进行分块处理
        """
        N, _, T, H, W = imgs.shape
        p = patch_size   #=2
        u = self.args.t_patch_size  #=2
        # assert H % p == 0 and W % p == 0 and T % u == 0
        h = H // p  #3
        w = W // p  #19
        t = T // u  #12
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

    def forward_encoder(self, x, x_mark, mask_ratio, mask_strategy, data=None, prompt={},patch_size=1,split_nodes=None):
        # embed patches
        N, _, T, H, W = x.shape     #torch.Size([204, 1, 24, 10, 20])

        # origin_x = x.clone()

        edges = prompt['topo']


        if self.args.isGraph:
            x, TimeEmb = self.Embedding_patch_graph(x, x_mark, edges, split_nodes, is_time=self.args.is_time_emb,patch_size=patch_size, hour_num=data)
        else:
            x, TimeEmb = self.Embedding_patch(x, x_mark, edges, is_time=self.args.is_time_emb, patch_size=patch_size,hour_num=data)


        _, L, C = x.shape

        T = T // self.args.t_patch_size



        x, mask, ids_restore, ids_keep = causal_masking(x, mask_ratio, T=T, mask_strategy=mask_strategy)

        if self.args.isGraph:
            input_size = (T, len(split_nodes), 1)
        else:
            input_size = (T, H // patch_size, W // patch_size)

        # 生成位置编码
        pos_embed_sort = self.pos_embed_enc(ids_keep, N, input_size)
        # assert x.shape == pos_embed_sort.shape

        # 加上位置编码
        x_attn = x + pos_embed_sort

        prompt_save = {}
        attn_bias = {}



        if self.args.is_prompt == 1 :

            prompt_t = self.enc_memory_t(prompt['t'].reshape(-1, prompt['t'].shape[-1]))
            prompt_t = prompt_t['out'].reshape(prompt['t'].shape)

            prompt_f = self.enc_memory_f(prompt['f'].reshape(-1, prompt['f'].shape[-1]))
            prompt_f = prompt_f['out'].reshape(prompt['f'].shape)

            adp_t = F.softmax(F.relu(prompt_t @ prompt_t.transpose(1, 2)), dim=-1)  # N * (H*W) * (H*W)
            adp_f = F.softmax(F.relu(prompt_f @ prompt_f.transpose(1, 2)), dim=-1)


            data_list_t = []
            data_list_f = []
            edge_att_t, edge_att_f = [], []

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


            if self.args.isGraph:
                prompt_t = self.gcn_topo_t(prompt_t.reshape(-1, prompt_t.shape[-1]), edge_t, edge_att_t).reshape(N,len(split_nodes),self.embed_dim)
                prompt_f = self.gcn_topo_f(prompt_f.reshape(-1, prompt_f.shape[-1]), edge_f, edge_att_f).reshape(N,len(split_nodes),self.embed_dim)
            else:
                prompt_t = self.gcn_t(prompt_t.reshape(-1, prompt_t.shape[-1]), edge_t, edge_att_t).reshape(N,H * W // patch_size ** 2,self.embed_dim)
                prompt_f = self.gcn_f(prompt_f.reshape(-1, prompt_f.shape[-1]), edge_f, edge_att_f).reshape(N,H * W // patch_size ** 2,self.embed_dim)


            prompt_t = prompt_t.unsqueeze(1).repeat(1, self.args.his_len // self.args.t_patch_size, 1, 1).reshape(N,x_attn.shape[1],self.embed_dim)
            prompt_f = prompt_f.unsqueeze(1).repeat(1, self.args.his_len // self.args.t_patch_size, 1, 1).reshape(N,x_attn.shape[1],self.embed_dim)


            # assert prompt_t.shape == prompt_f.shape == x_attn.shape

            prompt_save['t'] = prompt_t.clone()
            prompt_save['f'] = prompt_f.clone()

        if self.args.is_prompt == 1 :


            prompt_t = self.enc_memory_t(prompt['t'].reshape(-1, prompt['t'].shape[-1]))
            prompt_t = prompt_t['out'].reshape(prompt['t'].shape)

            prompt_f = self.enc_memory_f(prompt['f'].reshape(-1, prompt['f'].shape[-1]))
            prompt_f = prompt_f['out'].reshape(prompt['f'].shape)

            prompt_t = prompt_t.unsqueeze(1).repeat(1, self.args.his_len // self.args.t_patch_size, 1, 1).reshape(N,x_attn.shape[1],self.embed_dim)
            prompt_f = prompt_f.unsqueeze(1).repeat(1, self.args.his_len // self.args.t_patch_size, 1, 1).reshape(N,x_attn.shape[1],self.embed_dim)

            # assert prompt_t.shape == prompt_f.shape == x_attn.shape

            prompt_save['node_t'] = prompt_t.clone()
            prompt_save['node_f'] = prompt_f.clone()



        for index, blk in enumerate(self.enblocks):
            x_attn = blk(x_attn, attn_bias=attn_bias)

        return x_attn, mask, ids_restore, input_size, TimeEmb, prompt_save

    def forward_decoder(self, x, x_mark, mask, ids_restore, TimeEmb, input_size=None,
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

        if self.args.is_prompt == 1:
            prompt_t, prompt_f = prompt_graph['t'], prompt_graph['f']

            prompt_t = prompt_t.reshape(N, -1, H * W, prompt_t.shape[-1])[:, :1].repeat(1, (self.args.his_len + self.args.pred_len) // self.args.t_patch_size, 1, 1).reshape(N, -1,prompt_t.shape[-1])
            prompt_f = prompt_f.reshape(N, -1, H * W, prompt_f.shape[-1])[:, :1].repeat(1, (self.args.his_len + self.args.pred_len) // self.args.t_patch_size, 1, 1).reshape(N, -1,prompt_f.shape[-1])

            x_attn += prompt_f + prompt_t

            prompt_t, prompt_f = prompt_graph['node_t'], prompt_graph['node_f']
            prompt_t = prompt_t.reshape(N, -1, H * W, prompt_t.shape[-1])[:, :1].repeat(1, (self.args.his_len + self.args.pred_len) // self.args.t_patch_size, 1, 1).reshape(N, -1,prompt_t.shape[-1])
            prompt_f = prompt_f.reshape(N, -1, H * W, prompt_f.shape[-1])[:, :1].repeat(1, (self.args.his_len + self.args.pred_len) // self.args.t_patch_size, 1, 1).reshape(N, -1,prompt_f.shape[-1])

            x_attn += prompt_f + prompt_t


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

        return loss1, target

    def graph_loss(self, pred, target):

        loss1 = ((pred[:, self.args.his_len:] - target[:, self.args.his_len:]) ** 2).mean()

        loss2 = ((pred[:, :self.args.his_len] - target[:, :self.args.his_len]) ** 2).mean()

        mask = torch.ones_like(target)

        mask[:, :self.args.his_len] = 0

        return loss1, loss2, target, mask


    def adpative_graph(self, img, img_mark, DataEmbedding,node_split=None, patch_size=2):

        N, _, T, H, W = img.shape
        # img_mark : N * T * 2

        img_origin = img.clone().squeeze(dim=1).reshape(N, T, H * W)
        img_origin = img_origin.permute(0, 2, 1).reshape(N * H * W, T)  # (N*H*W) * T
        img_origin = img_origin[:, :self.args.his_len]    # only use history data

        img_spec = torch.fft.rfft(img_origin, n=img_origin.shape[-1], norm="ortho",dim=-1)  # [N, K] K = T//2 + 1
        img_spec = img_spec.reshape(N, H, W, self.args.his_len // 2 + 1)

        img_spec = torch.cat((img_spec.real, img_spec.imag), dim=-1)

        img_spec = self.spec_liner(img_spec)

        img_tmp = img_origin.unsqueeze(1)

        img_tmp = self.temporaltokenConv(img_tmp).permute(0, 2, 1)  # N * T * Embed

        img_mark = img_mark[:, :self.args.his_len].unsqueeze(dim=1).repeat(1, H * W, 1, 1).reshape(N * H * W,self.args.his_len,2)

        temporal_emb = DataEmbedding.temporal_emb(img_mark, self.args.dataset)

        img_tmp += temporal_emb


        img_tmp = torch.cat([self.temporal_attn_encoder(img_tmp[index:index + H * W]) for index in range(0, img_tmp.shape[0], H * W)],axis=0)[:, 0]


        img_tmp = img_tmp.reshape(N, H, W, img_tmp.shape[-1])



        if self.args.isGraph:
            # patchify
            max_len = max([len(i) for i in node_split])
            n_group = len(node_split)

            img_tmp = torch.cat([torch.mean(torch.gather(img_tmp, 1,group.view(1, group.shape[0], 1, 1).expand(img_tmp.shape[0],group.shape[0],img_tmp.shape[2],img_tmp.shape[3]).to(img_tmp).long()), dim=1, keepdim=True) for group in node_split], dim=1).squeeze(dim=2)

            img_spec = torch.cat([torch.mean(torch.gather(img_spec, 1,group.view(1, group.shape[0], 1, 1).expand(img_spec.shape[0],group.shape[0],img_spec.shape[2],img_spec.shape[3]).to(img_tmp).long()), dim=1, keepdim=True) for group in node_split], dim=1).squeeze(dim=2)

        else:
            if patch_size == 1:
                img_spec = self.prompt_spatial_patch_f_1(img_spec.permute(0, 3, 1, 2))
                img_tmp = self.prompt_spatial_patch_t_1(img_tmp.permute(0, 3, 1, 2))
            elif patch_size == 2:
                img_spec = self.prompt_spatial_patch_f_2(img_spec.permute(0, 3, 1, 2))
                img_tmp = self.prompt_spatial_patch_t_2(img_tmp.permute(0, 3, 1, 2))
            elif patch_size == 5:
                img_spec = self.prompt_spatial_patch_f_5(img_spec.permute(0, 3, 1, 2))
                img_tmp = self.prompt_spatial_patch_t_5(img_tmp.permute(0, 3, 1, 2))


        return img_tmp, img_spec


    # Create target key padding mask
    def create_padding_mask(self, seq_lengths, max_len):
        padding_mask = torch.zeros((len(seq_lengths), max_len), dtype=torch.bool)
        for i, length in enumerate(seq_lengths):
            padding_mask[i, length:] = True
        return padding_mask

    def forward(self,X,Y, timestamps, topo,subgraphs=None):
        #模型输入输出:[32, 228, 1, 12]
        B=X.shape[0]
        value = torch.cat([X, Y], dim=3)
        value = value.permute(0, 2, 3, 1)
        value=value.reshape(B,1,24,self.args.wide, self.args.height)   #pemsd7:6  38  |pems08 10 17



        patch_size = self.args.patch_size  #2
        if self.args.is_prompt == 1:
            if self.args.isGraph:

                adj_dynamic, S_soft, subgraphs = self.graph_partition()
                # 图结构一致性损失（MinCut思路）
                A = adj_dynamic
                S = S_soft
                A_pool = torch.mm(S.T, torch.mm(A, S))
                loss_cut = -torch.trace(A_pool) / torch.sum(A_pool)  # 最小割约束
                # 平衡性约束（防止所有节点落入同一簇）
                loss_balance = torch.mean((torch.sum(S, dim=0) - A.shape[0] / S.shape[1]) ** 2)

                img_tmp, img_spec = self.adpative_graph(value, timestamps, self.Embedding_patch_graph, node_split = subgraphs, patch_size = patch_size)
            else:
                img_tmp, img_spec = self.adpative_graph(value, timestamps, self.Embedding_patch,patch_size=patch_size)  # 时域/频域
        else:
            img_tmp = None
            img_spec = None


        T, H, W = value.shape[2:]

        latent, mask, ids_restore, input_size, TimeEmb, prompt = self.forward_encoder(value, timestamps, self.args.mask_ratio,mask_strategy=self.args.mask_strategy,data=self.args.dataset,prompt={'t': img_tmp,'f': img_spec,'topo': topo},patch_size=patch_size,split_nodes=subgraphs)

        #latent, mask, ids_restore, input_size, TimeEmb, prompt = self.forward_encoder(value, timestamps, mask_ratio=self.args.mask_ratio,mask_strategy=self.args.mask_strategy,prompt={'t': img_tmp,'f': img_spec,'topo': topo},patch_size=patch_size)

        pred = self.forward_decoder(latent, timestamps, mask, ids_restore, TimeEmb, input_size=input_size,prompt_graph=prompt)  # [N, L, p*p*1]

        L = pred.shape[1]
        perm=None


        if self.args.isGraph:

            seq_lengths = [len(i) for i in subgraphs]
            max_len = max(seq_lengths)
            pred=self.pred_model_linear_GraphPems(pred).reshape(pred.shape[0], T // self.args.t_patch_size,len(subgraphs), self.args.t_patch_size, -1).permute(0, 1, 3, 2, 4)


            pred = pred.reshape(pred.shape[0], T, len(subgraphs), -1)

            pred = torch.cat([pred[:, :, g, :seq_lengths[g]] for g in range(pred.shape[2])], dim=2)

            #torch.Size([3, 24, 21073])

            target = value.squeeze(dim=(1, 4))
            target = torch.cat([torch.gather(target, 2,group.view(1, 1, group.shape[0]).expand(target.shape[0], target.shape[1],group.shape[0]).to(target).long())for group in subgraphs], dim=2)

            loss1, loss2, target, mask = self.graph_loss(pred, target)
            loss_total = loss1 + 0.1 * loss_cut + 0.2 * loss_balance
        else:
            if patch_size == 1:
                pred = self.head_layer_1(pred)
            elif patch_size == 2:
                pred = self.head_layer_2(pred)
            elif patch_size == 4:
                pred = self.head_layer_4(pred)
            elif patch_size == 5:
                pred = self.head_layer_5(pred)
            loss_total, target = self.forward_loss(value, pred, mask, patch_size)


        return loss_total, pred, target, mask,perm

