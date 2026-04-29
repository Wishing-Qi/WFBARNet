from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hd_dim: int, drop_p: float = 0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hd_dim),
            nn.GELU(),
            nn.Dropout(drop_p, inplace=True),
            nn.Linear(hd_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class MLP_Head(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hd_dim: int, drop_p: float = 0.0) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, out_dim, hd_dim, drop_p)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self.layer_norm(x))


class FeedForward(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hd_dim: int, drop_p: float = 0.0) -> None:
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, hd_dim, drop_p)
        self.dropout = nn.Dropout(drop_p, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.mlp(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_head: int, n_head: int, drop_p: float) -> None:
        super().__init__()
        d_cat = d_head * n_head
        self.h = n_head
        self.to_qkv = nn.Linear(d_model, d_cat * 3, bias=False)
        self.scale = d_head**-0.5
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(drop_p))
        self.tail = (
            nn.Sequential(nn.Linear(d_cat, d_model), nn.Dropout(drop_p, inplace=True))
            if n_head != 1 or d_cat != d_model
            else nn.Identity()
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_n, steps, _ = x.shape
        qkv = self.to_qkv(x).view(batch_n, steps, self.h, -1).chunk(3, dim=-1)
        q, k, v = (part.transpose(1, 2) for part in qkv)
        dots = (q.contiguous() @ k.transpose(-1, -2).contiguous()) * self.scale
        if mask is not None:
            dots = dots.masked_fill(mask.view(batch_n, 1, 1, steps) == 0.0, -torch.inf)
        attended = self.attend(dots) @ v.contiguous()
        out = attended.transpose(1, 2).reshape(batch_n, steps, -1)
        return self.tail(out)


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, d_head: int, n_head: int, hd_mlp: int, drop_p: float) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, d_head, n_head, drop_p)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_model, hd_mlp, drop_p)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.attn(self.layer_norm1(x), mask) + x
        return self.ff(self.layer_norm2(x)) + x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, d_head: int, n_head: int, depth: int, hd_mlp: int, drop_p: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, d_head, n_head, hd_mlp, drop_p) for _ in range(depth)]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TCN(nn.Module):
    def __init__(self, in_channel: int, channels: list[int], kernel_size: int = 5, drop_p: float = 0.3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index, out_ch in enumerate(channels):
            in_ch = in_channel if index == 0 else channels[index - 1]
            dilation = index * 2 + 1
            padding = (kernel_size - 1) * dilation // 2
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                    nn.Dropout(drop_p, inplace=True),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, d_head: int, n_head: int, drop_p: float) -> None:
        super().__init__()
        d_cat = d_head * n_head
        self.h = n_head
        self.to_q = nn.Linear(d_model, d_cat, bias=False)
        self.to_kv = nn.Linear(d_model, d_cat * 2, bias=False)
        self.scale = d_head**-0.5
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(drop_p))
        self.tail = (
            nn.Sequential(nn.Linear(d_cat, d_model), nn.Dropout(drop_p, inplace=True))
            if n_head != 1 or d_cat != d_model
            else nn.Identity()
        )

    def forward(self, x1: Tensor, x2: Tensor, mask: Tensor | None = None) -> Tensor:
        query = self.to_q(x1)
        key_value = self.to_kv(x2)
        batch, steps, _ = query.shape
        query = query.view(batch, steps, self.h, -1).transpose(1, 2)
        key, value = (part.transpose(1, 2) for part in key_value.view(batch, steps, self.h, -1).chunk(2, dim=-1))
        dots = (query.contiguous() @ key.transpose(-1, -2).contiguous()) * self.scale
        if mask is not None:
            dots = dots.masked_fill(mask.view(batch, 1, 1, steps) == 0.0, -torch.inf)
        attended = self.attend(dots) @ value.contiguous()
        out = attended.transpose(1, 2).reshape(batch, steps, -1)
        return self.tail(out)


class CrossTransformerLayer(nn.Module):
    def __init__(self, d_model: int, d_head: int, n_head: int, hd_mlp: int, drop_p: float) -> None:
        super().__init__()
        self.layer_norm1_x1 = nn.LayerNorm(d_model)
        self.layer_norm1_x2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadCrossAttention(d_model, d_head, n_head, drop_p)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_model, hd_mlp, drop_p)

    def forward(self, x1: Tensor, x2: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.cross_attn(self.layer_norm1_x1(x1), self.layer_norm1_x2(x2), mask)
        return self.ff(self.layer_norm2(x)) + x


def _sinusoidal_like(param: Tensor) -> Tensor:
    steps = param.shape[-2]
    dims = param.shape[-1]
    pe = torch.zeros(steps, dims, dtype=param.dtype, device=param.device)
    position = torch.arange(steps, dtype=param.dtype, device=param.device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dims, 2, dtype=param.dtype, device=param.device) * (-math.log(10000.0) / dims))
    pe[:, 0::2] = torch.sin(position * div_term)
    if dims > 1:
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.view(*((1,) * (param.ndim - 2)), steps, dims).expand_as(param)


class _BSTBase(nn.Module):
    has_position_fusion = False
    has_clean_gate = False
    has_aim_player = False

    def __init__(
        self,
        in_dim: int,
        seq_len: int,
        n_class: int = 35,
        n_people: int = 2,
        d_model: int = 100,
        d_head: int = 128,
        n_head: int = 6,
        depth_tem: int = 2,
        depth_inter: int = 1,
        drop_p: float = 0.3,
        mlp_d_scale: int = 4,
        tcn_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if n_people > 2:
            raise NotImplementedError("Official BST supports at most two people.")
        self.d_model = d_model
        if self.has_position_fusion:
            self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)
        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, [d_model // 2, d_model], tcn_kernel_size, drop_p)
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        if self.has_aim_player:
            self.cos_sim = nn.CosineSimilarity()
        if self.has_clean_gate:
            self.mlp_clean = MLP(d_model, d_model, d_model, drop_p)
        head_in_dim = d_model * 2 if self.has_aim_player and not self.has_clean_gate else d_model * 3
        self.mlp_head = MLP_Head(head_in_dim, n_class, d_model * mlp_d_scale, drop_p)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        self.embedding_tem.copy_(_sinusoidal_like(self.embedding_tem))
        self.embedding_cross.copy_(_sinusoidal_like(self.embedding_cross))
        self.embedding_inter.copy_(_sinusoidal_like(self.embedding_inter))
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)
        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv1d):
            nn.init.xavier_normal_(module.weight)

    def _forward_core(
        self,
        jnb: Tensor,
        shuttle: Tensor,
        video_len: Tensor,
        pos: Tensor | None = None,
    ) -> Tensor:
        batch, steps, people, in_dim = jnb.shape
        jnb = jnb.permute(0, 2, 3, 1).reshape(batch * people, in_dim, steps)
        if self.has_position_fusion:
            if pos is None:
                raise ValueError("This BST variant requires player positions.")
            pos_impact = self.mlp_positions(pos).permute(0, 2, 3, 1).reshape(batch * people, in_dim, steps)
            jnb = jnb * pos_impact + jnb
        jnb = self.tcn_pose(jnb).view(batch, people, -1, steps).transpose(-2, -1)
        shuttle = self.tcn_shuttle(shuttle.transpose(1, 2).contiguous()).unsqueeze(1).transpose(-2, -1)
        x = torch.cat((jnb, shuttle), dim=1)
        _, streams, _, hidden = x.shape
        class_token_tem = self.learned_token_tem.view(1, 1, -1).expand(batch * streams, -1, -1)
        x = x.view(batch * streams, steps, hidden)
        x = torch.cat((class_token_tem, x), dim=1) + self.embedding_tem
        range_t = torch.arange(0, 1 + steps, device=x.device).unsqueeze(0).expand(batch, -1)
        mask = range_t < (1 + video_len.unsqueeze(-1))
        mask_stream = mask.repeat_interleave(streams, dim=0)
        x = self.encoder_tem(self.pre_dropout(x), mask_stream).view(batch, streams, 1 + steps, hidden)
        p1, p2, shuttle_latent = (part.squeeze(1) for part in x.chunk(3, dim=1))
        p1_cls = p1[:, 0].contiguous()
        p2_cls = p2[:, 0].contiguous()
        shuttle_cls = shuttle_latent[:, 0].contiguous()
        p1 = p1[:, 1:].contiguous() + self.embedding_cross
        p2 = p2[:, 1:].contiguous() + self.embedding_cross
        shuttle_latent = shuttle_latent[:, 1:].contiguous() + self.embedding_cross
        cross_mask = mask[:, 1:].contiguous()
        p1_shuttle = self.cross_trans(p1, shuttle_latent, cross_mask)
        p2_shuttle = self.cross_trans(p2, shuttle_latent, cross_mask)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(batch, -1, -1)
        p1_shuttle = torch.cat((class_token_inter, p1_shuttle), dim=1) + self.embedding_inter
        p2_shuttle = torch.cat((class_token_inter, p2_shuttle), dim=1) + self.embedding_inter
        p1_shuttle = self.encoder_inter(p1_shuttle, mask)
        p2_shuttle = self.encoder_inter(p2_shuttle, mask)
        p1_shuttle_cls = p1_shuttle[:, 0, :].contiguous()
        p2_shuttle_cls = p2_shuttle[:, 0, :].contiguous()
        p1_conclusion = p1_cls + p1_shuttle_cls
        p2_conclusion = p2_cls + p2_shuttle_cls
        if self.has_aim_player:
            alpha = (self.cos_sim(p1_shuttle_cls, shuttle_cls) - self.cos_sim(p2_shuttle_cls, shuttle_cls) + 2) / 4
            alpha = alpha.unsqueeze(1)
            p1_conclusion = alpha * p1_conclusion
            p2_conclusion = (1 - alpha) * p2_conclusion
        if self.has_clean_gate:
            dirt = self.mlp_clean(torch.minimum(p1_shuttle_cls, p2_shuttle_cls))
            shuttle_cls = shuttle_cls - dirt
        if self.has_aim_player and not self.has_clean_gate:
            x = torch.cat((p1_conclusion, p2_conclusion), dim=1)
        else:
            x = torch.cat((p1_conclusion, p2_conclusion, shuttle_cls), dim=1)
        return self.mlp_head(x)


class BST_0(_BSTBase):
    def forward(self, JnB: Tensor, shuttle: Tensor, video_len: Tensor) -> Tensor:
        return self._forward_core(JnB, shuttle, video_len)


class BST(_BSTBase):
    has_position_fusion = True

    def forward(self, JnB: Tensor, shuttle: Tensor, pos: Tensor, video_len: Tensor) -> Tensor:
        return self._forward_core(JnB, shuttle, video_len, pos)


class BST_CG(_BSTBase):
    has_position_fusion = True
    has_clean_gate = True

    def forward(self, JnB: Tensor, shuttle: Tensor, pos: Tensor, video_len: Tensor) -> Tensor:
        return self._forward_core(JnB, shuttle, video_len, pos)


class BST_AP(_BSTBase):
    has_position_fusion = True
    has_aim_player = True

    def forward(self, JnB: Tensor, shuttle: Tensor, pos: Tensor, video_len: Tensor) -> Tensor:
        return self._forward_core(JnB, shuttle, video_len, pos)


class BST_CG_AP(_BSTBase):
    has_position_fusion = True
    has_clean_gate = True
    has_aim_player = True

    def forward(self, JnB: Tensor, shuttle: Tensor, pos: Tensor, video_len: Tensor) -> Tensor:
        return self._forward_core(JnB, shuttle, video_len, pos)
