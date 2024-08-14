# ---------------------------------------------
#  Modified by Yuqing Wen
# ------------------------------------------------------------------------
# Copyright (c) 2023 Stability AI. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from (https://github.com/Stability-AI/generative-models)
# Copyright (c) 2023 Stability AI
# ------------------------------------------------------------------------

import math
from inspect import isfunction
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from packaging import version
from torch import nn

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}
    print(
        f"No SDP backend available, likely because you are running in pytorch versions < 2.0. In fact, "
        f"you are using PyTorch {torch.__version__}. You might want to consider upgrading."
    )

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("no module 'xformers'. Processing without...")

from .diffusionmodules.util import checkpoint
import warnings

def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context) ### take care about the dim of context
        v = self.to_v(context)
        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        ## old
        """
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        """
        ## new
        with sdp_kernel(**BACKEND_MAP[self.backend]):
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
    ):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads with a dimension of {dim_head}."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class MemoryEfficientIntraViewAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
    ):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads with a dimension of {dim_head}."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)


        context = default(context, x)
        k_all = self.to_k(context)
        v_all = self.to_v(context)

        q_all=self.to_q(x)

        H=int(math.sqrt(x.shape[1]//12))

        q_all = rearrange(q_all, "b (h w) c -> b c h w", h=H).contiguous()

        _, _, H, W = q_all.shape
        k_all = rearrange(k_all, "b (h w) c -> b c h w",h=H).contiguous()
        v_all = rearrange(v_all, "b (h w) c -> b c h w",h=H).contiguous()
        out_all=[]
        for i in range(0, W, int(W//6)):
            q = q_all[:, :, :, i:i+int(W//6)]
            q = rearrange(q, "b c h w -> b (h w) c").contiguous()
            k = k_all[:, :, :, i:i+int(W//6)]
            k = rearrange(k, "b c h w -> b (h w) c").contiguous()
            v = v_all[:, :, :, i:i+int(W // 6)]
            v = rearrange(v, "b c h w -> b (h w) c").contiguous()

            if n_times_crossframe_attn_in_self:
                # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
                assert x.shape[0] % n_times_crossframe_attn_in_self == 0
                # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
                k = repeat(
                    k[::n_times_crossframe_attn_in_self],
                    "b ... -> (b n) ...",
                    n=n_times_crossframe_attn_in_self,
                )
                v = repeat(
                    v[::n_times_crossframe_attn_in_self],
                    "b ... -> (b n) ...",
                    n=n_times_crossframe_attn_in_self,
                )
            b, _, _ = q.shape
            q, k, v = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (q, k, v),
            )

            # actually compute the attention, what we cannot get enough of
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )

            # TODO: Use this directly in the attention operation, as a bias
            if exists(mask):
                raise NotImplementedError
            out = (
                out.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
            )
            if additional_tokens is not None:
                # remove additional token
                out = out[:, n_tokens_to_mask:]
            out = rearrange(out, "b (h w) c -> b c h w", h=H).contiguous()
            out_all.append(out)
        out=torch.cat(out_all , dim=-1)
        out = rearrange(out, "b c h w -> b (h w) c").contiguous()
        return self.to_out(out)



class MemoryEfficientInterViewAttentionTwo(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
    ):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads with a dimension of {dim_head}."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        context = default(context, x)
        k_all = self.to_k(context)
        v_all = self.to_v(context)
        q_all=self.to_q(x)

        H=int(math.sqrt(x.shape[1]//12))

        q_all = rearrange(q_all, "b (h w) c -> b c h w", h=H).contiguous()
        _, _, H, W = q_all.shape
        k_all = rearrange(k_all, "b (h w) c -> b c h w",h=H).contiguous()
        v_all = rearrange(v_all, "b (h w) c -> b c h w",h=H).contiguous()
        out_all=[]
        width = int(W//6)
        for i in range(0, W, int(W//6)):
            q = q_all[:, :, :, i:i+int(W//6)]
            q = rearrange(q, "b c h w -> b (h w) c").contiguous()

            if i>0 and i < 6*width:
                k = torch.cat([k_all[:, :, :, i-width:i],k_all[:, :, :, i+width:i+2*width]],dim=-1)
                v = torch.cat([v_all[:, :, :, i-width:i],v_all[:, :, :, i+width:i+2*width]],dim=-1)
                # print(" use two 1-4",k.shape)
            elif i == 0:
                k = torch.cat([k_all[:, :, :, 5*width:W],k_all[:, :, :, i+width:i+2*width]],dim=-1)
                v = torch.cat([v_all[:, :, :, 5*width:W],v_all[:, :, :, i+width:i+2*width]],dim=-1)
                # print(" use two 0",k.shape)
            elif i == 6*width:
                k = torch.cat([k_all[:, :, :, i-width:i],k_all[:, :, :, 0:width]],dim=-1)
                v = torch.cat([v_all[:, :, :, i-width:i],v_all[:, :, :, 0:width]],dim=-1)
                # print(" use two 5",k.shape)

            k = rearrange(k, "b c h w -> b (h w) c").contiguous()
            v = rearrange(v, "b c h w -> b (h w) c").contiguous()

            if n_times_crossframe_attn_in_self:
                # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
                assert x.shape[0] % n_times_crossframe_attn_in_self == 0
                # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
                k = repeat(
                    k[::n_times_crossframe_attn_in_self],
                    "b ... -> (b n) ...",
                    n=n_times_crossframe_attn_in_self,
                )
                v = repeat(
                    v[::n_times_crossframe_attn_in_self],
                    "b ... -> (b n) ...",
                    n=n_times_crossframe_attn_in_self,
                )
            b, _, _ = q.shape
            q, k, v = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (q, k, v),
            )

            # actually compute the attention, what we cannot get enough of
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )

            # TODO: Use this directly in the attention operation, as a bias
            if exists(mask):
                raise NotImplementedError
            out = (
                out.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
            )
            if additional_tokens is not None:
                # remove additional token
                out = out[:, n_tokens_to_mask:]
            out = rearrange(out, "b (h w) c -> b c h w", h=H).contiguous()
            out_all.append(out)
        out=torch.cat(out_all , dim=-1)
        out = rearrange(out, "b c h w -> b (h w) c").contiguous()
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,  # ampere
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attn_mode="softmax",
        sdp_backend=None,
        temporal_transformer_attn_type=None,
        spatial_only_attn_type=None,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            print(
                f"Attention mode '{attn_mode}' is not available. Falling back to native attention. "
                f"This is not a problem in Pytorch >= 2.0. FYI, you are running with PyTorch version {torch.__version__}"
            )
            attn_mode = "softmax"
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            print(
                "We do not support vanilla attention anymore, as it is too expensive. Sorry."
            )
            if not XFORMERS_IS_AVAILABLE:
                assert (
                    False
                ), "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                print("Falling back to xformers efficient attention.")
                attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        self.disable_self_attn = disable_self_attn
        self.temporal_transformer_attn_type = temporal_transformer_attn_type
        self.spatial_only_attn_type=spatial_only_attn_type

        if spatial_only_attn_type == 'intra-view':
            self.attn1 = MemoryEfficientIntraViewAttention(
                query_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                context_dim=context_dim if self.disable_self_attn else None,
                backend=sdp_backend,
            )
        elif spatial_only_attn_type == 'inter-view':
            self.attn1 = MemoryEfficientInterViewAttentionTwo(
                query_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                context_dim=context_dim if self.disable_self_attn else None,
                backend=sdp_backend,
            )
        else:
            self.attn1 = attn_cls(
                query_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                context_dim=context_dim if self.disable_self_attn else None,
                backend=sdp_backend,
            )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        self.attn2 = attn_cls(
        query_dim=dim,
        context_dim=context_dim,
        heads=n_heads,
        dim_head=d_head,
        dropout=dropout,
        backend=sdp_backend,
    )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        kwargs = {"x": x}

        if context is not None:
            kwargs.update({"context": context})

        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        if n_times_crossframe_attn_in_self:
            kwargs.update(
                {"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self}
            )

        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        
        x = (
            self.attn1(
            self.norm1(x), context=context if self.disable_self_attn else None, additional_tokens=additional_tokens,
            n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
        if not self.disable_self_attn
        else 0,
            )
        + x
        )

        x = (
            self.attn2(
                self.norm2(x), context=context, additional_tokens=additional_tokens
            )
            + x
        )
        x = self.ff(self.norm3(x)) + x
        return x


class BasicTransformerSingleLayerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention  # on the A100s not quite as fast as the above version
        # (todo might depend on head_dim, check, falls back to semi-optimized kernels for dim!=[16,32,64,128])
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        attn_mode="softmax",
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context) + x
        x = self.ff(self.norm2(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax",
        use_checkpoint=True,
        # sdp_backend=SDPBackend.FLASH_ATTENTION
        sdp_backend=None,
    ):
        super().__init__()
        print(
            f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads"
        )
        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                print(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

class SpatialTemporalTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax",
        use_checkpoint=True,
        # sdp_backend=SDPBackend.FLASH_ATTENTION
        sdp_backend=None,
        alpha=1,
        num_frames=4,
        temporal_transformer_attn_type=None,
        spatial_only_attn_type=None,
        insert_crossview=False,
    ):
        super().__init__()
        self.insert_crossview=insert_crossview
        self.num_frames=num_frames
        print(
            f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads"
        )
        from omegaconf import ListConfig
        self.alpha = 1 if alpha == 1 else nn.Parameter(torch.rand(1, requires_grad=True))
        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                print(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        ##add temporal layer   project in not zero initialized
        self.norm_temporal = Normalize(in_channels)

        if not use_linear:
            self.proj_in_temporal =nn.Conv1d(
                    in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in_temporal = nn.Linear(in_channels, inner_dim)


        if self.insert_crossview:
            ##insert cross view
            self.norm_crossview = Normalize(in_channels)
            if not use_linear:
                self.proj_in_crossview = nn.Conv1d(
                        in_channels, inner_dim, kernel_size=1, stride=1, padding=0
                    )
            else:
                self.proj_in_crossview = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                    spatial_only_attn_type=spatial_only_attn_type,
                )
                for d in range(depth)
            ]
        )
        self.transformer_blocks_temporal = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                    temporal_transformer_attn_type=temporal_transformer_attn_type,
                )
                for d in range(depth)
            ]
        )
        if self.insert_crossview:
            assert spatial_only_attn_type=='intra-view'
            self.transformer_blocks_crossview = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        inner_dim,
                        n_heads,
                        d_head,
                        dropout=dropout,
                        context_dim=context_dim[d],
                        disable_self_attn=disable_self_attn,
                        attn_mode=attn_type,
                        checkpoint=use_checkpoint,
                        sdp_backend=sdp_backend,
                        spatial_only_attn_type='inter-view',
                    )
                    for d in range(depth)
                ]
            )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        ## add temporal
        if not use_linear:
            self.proj_out_temporal = zero_module(
                nn.Conv1d(
                    inner_dim, in_channels, kernel_size=1, stride=1, padding=0
                )
            )
        else:
            self.proj_out_temporal = zero_module(nn.Linear(inner_dim, in_channels))

        if self.insert_crossview:
            if not use_linear:
                self.proj_out_crossview = zero_module(
                    nn.Conv1d(
                        inner_dim, in_channels, kernel_size=1, stride=1, padding=0
                    )
                )
            else:
                self.proj_out_crossview = zero_module(nn.Linear(inner_dim, in_channels))

        self.use_linear = use_linear
        self.identity_layer = nn.Identity()

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        x = x + x_in

        if self.insert_crossview:
            x_in = x
            x = self.norm_crossview(x)
            if not self.use_linear:
                x = self.proj_in_crossview(x)
            x = rearrange(x, "b c h w -> b (h w) c").contiguous()
            if self.use_linear:
                x = self.proj_in_crossview(x)
            for i, block in enumerate(self.transformer_blocks_crossview):
                if i > 0 and len(context) == 1:
                    i = 0  # use same context for each block
                x = block(x, context=context[i])
            if self.use_linear:
                x = self.proj_out_crossview(x)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
            if not self.use_linear:
                x = self.proj_out_crossview(x)
            x = x + x_in

        ## add temporal
        x_in = x
        x = self.norm_temporal(x)
        if not self.use_linear:
            x = rearrange(x, "(b t) c h w -> (b h w) c t",t=self.num_frames).contiguous() ## TO DO
            x = self.proj_in_temporal(x)
            x = rearrange(x, "(b h w) c t -> (b t) c h w", h=h, w=w).contiguous()
        x = rearrange(x, "b c h w -> b (h w) c").contiguous() ##here b is actually (b t)
        if self.use_linear:
            x = self.proj_in_temporal(x) ## x (b t) (h w) c
        x = rearrange(x, "(b t) (h w) c -> (b h w) t c", t=self.num_frames, h=h, w=w).contiguous()
        pos_embed_temporal = create_1d_absolute_sin_cos_embedding(self.num_frames, c).to(x)
        x = x + pos_embed_temporal
        for i, block in enumerate(self.transformer_blocks_temporal):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            contexti = rearrange(context[i], "(b t) n c -> b t n c ", t=self.num_frames).contiguous()[:, 0:1]
            contexti = contexti.unsqueeze(1).repeat(1, h, w, 1, 1)
            contexti = rearrange(contexti, "b h w n c -> (b h w) n c ").contiguous()
            x = block(x, context=contexti)  ## To do, how to merge context??
            #x = block(x)
        x = rearrange(x, "(b h w) t c -> (b t) (h w) c", h=h, w=w).contiguous()
        if self.use_linear:
            x = self.proj_out_temporal(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous() ##here b is actually (b t)
        if not self.use_linear:
            x = rearrange(x, "(b t) c h w -> (b h w) c t", t=self.num_frames).contiguous()  ## TO DO
            x = self.proj_out_temporal(x)  ## x (b t) (h w) c
        return x_in + x

import torch


# 1d绝对sin_cos编码
def create_1d_absolute_sin_cos_embedding(pos_len, dim):
    assert dim % 2 == 0, "wrong dimension!"
    position_emb = torch.zeros(pos_len, dim, dtype=torch.float)
    # i矩阵
    i_matrix = torch.arange(dim//2, dtype=torch.float)
    i_matrix /= dim / 2
    i_matrix = torch.pow(10000, i_matrix)
    i_matrix = 1 / i_matrix
    i_matrix = i_matrix.to(torch.long)
    # pos矩阵
    pos_vec = torch.arange(pos_len).to(torch.long)
    # 矩阵相乘，pos变成列向量，i_matrix变成行向量
    out = pos_vec[:, None] @ i_matrix[None, :]
    # 奇/偶数列
    emb_cos = torch.cos(out)
    emb_sin = torch.sin(out)
    # 赋值
    position_emb[:, 0::2] = emb_sin
    position_emb[:, 1::2] = emb_cos
    return position_emb
