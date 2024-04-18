#  Modified by Yuqing Wen
# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
from torch.nn.init import (
    xavier_uniform_,
    constant_,
    xavier_normal_
)
from torch.nn.functional import linear

from einops import rearrange
from mmcv.runner import auto_fp16
from mmcv.runner.base_module import BaseModule



def _in_projection_packed(q, k, v, w, b = None):
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

