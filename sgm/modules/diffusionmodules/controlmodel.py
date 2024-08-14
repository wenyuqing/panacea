import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...modules.diffusionmodules.util import (
    conv_nd,
    timestep_embedding,
    zero_module,
    linear,
)
from ...util import default, exists, instantiate_from_config

from sgm.modules.diffusionmodules.openaimodel import (
    TimestepEmbedSequential,
    UNetModel3D,
)

class ControlNet3D(UNetModel3D):
    """A locked copy branch of UNetModel3D that processes task-specific conditions.
    The model weights are initilized from the weights of the pretrained UNetModel3D.
    The additional input_hint_block is used to transform the input condition into the
    same dimension as the output of the vae-encoder
    """

    def __init__(
        self, hint_channels, control_scales, dims=2,disable_temporal=False, *args, **kwargs
    ):
        kwargs["out_channels"] = kwargs["in_channels"]  
        self.control_scales = control_scales
        self.hint_channels = hint_channels
        self.disable_temporal = disable_temporal
        super().__init__(*args, **kwargs)

        model_channels = kwargs["model_channels"]
        channel_mult = kwargs["channel_mult"]
        del self.output_blocks
        del self.out
        if hasattr(self, "id_predictor"):
            del self.id_predictor
            del self.id_predictor_temporal

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )
        # this is for the transformation of hint
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels, dims=2)])


        input_block_chans = [model_channels]
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                ch = mult * model_channels
                self.zero_convs.append(self.make_zero_conv(ch, dims=2))

                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.zero_convs.append(self.make_zero_conv(ch, dims=2))

        self.middle_block_out = self.make_zero_conv(ch, dims=2)

        if disable_temporal:
            self.setup_disbale_temporal()


    def make_zero_conv(self, channels, dims=2):
        return TimestepEmbedSequential(
            zero_module(conv_nd(dims, channels, channels, 1, padding=0))
        )

    def forward(self, x, hint, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb=t_emb.to(self.input_hint_block[0].weight.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            y = y.unsqueeze(1).repeat(1, self.num_frames,  1)
            y = rearrange(y, "b t c -> (b t) c ").contiguous()
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)
        if self.hint_channels > 19: #hint 8 19 256 3072 #
            if self.training:
                hint = rearrange(hint, "(t m) c h w -> t m c h w ",t=self.num_frames).contiguous()
                hint = rearrange(hint, "t m c h w -> t (m c) h w ").contiguous()
            else:
                hint_x1 = rearrange(hint[:8], "(t m) c h w -> t m c h w ",t=self.num_frames).contiguous()
                hint_x1 = rearrange(hint_x1, "t m c h w -> t (m c) h w ").contiguous()
                hint_x2 = rearrange(hint[8:], "(t m) c h w -> t m c h w ",t=self.num_frames).contiguous()
                hint_x2 = rearrange(hint_x2, "t m c h w -> t (m c) h w ").contiguous()
                hint = th.cat([hint_x1,hint_x2])
        guided_hint = self.input_hint_block(hint, emb, context)
        outs = []

        context = context.unsqueeze(1).repeat(1,self.num_frames,1,1)
        context = rearrange(context, "b t n c -> (b t) n c ").contiguous()

        h = x
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context) ##minist: x 512 1 28 28 emb: 512 128 c:
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        control_scales = [self.control_scales for _ in range(len(outs))]
        control = [
            c * scale for c, scale in zip(outs, control_scales)
        ]  # Adjusting the strength of control

        return control



class ControlledUNetModel3D(UNetModel3D):
    """A trainable copy branch of UNetModel3D that processes the video inputs.
    The model weights are initilized from the weights of the pretrained UNetModel3D.
    """

    def __init__(
        self, controlnet_config=None, only_add_on_center_frame=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if controlnet_config is not None:
            self.controlnet = instantiate_from_config(controlnet_config)
        


    def forward(self, x, timesteps=None, context=None, y=None, control=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = t_emb.to(self.input_blocks[0][0].weight.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            y = y.unsqueeze(1).repeat(1, self.num_frames,  1)
            y = rearrange(y, "b t c -> (b t) c ").contiguous()
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        context = context.unsqueeze(1).repeat(1,self.num_frames,1,1)
        context = rearrange(context, "b t n c -> (b t) n c ").contiguous()
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context) ##minist: x 512 1 28 28 emb: 512 128 c:
            hs.append(h)
        h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()
        for i, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)

        if self.predict_codebook_ids:
            assert False, "not supported anymore. what the f*** are you doing?"
        else:
            return self.out(h)
