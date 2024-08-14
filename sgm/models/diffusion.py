# ---------------------------------------------
#  Modified by Yuqing Wen
# ------------------------------------------------------------------------
# Copyright (c) 2023 Stability AI. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from (https://github.com/Stability-AI/generative-models)
# Copyright (c) 2023 Stability AI
# ------------------------------------------------------------------------

from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
from omegaconf import ListConfig, OmegaConf
from ..modules import UNCONDITIONAL_CONFIG
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER,OPENAIUNETWRAPPERCONTROLLDM3D
from sgm.modules.encoders.modules import VAEEmbedder
from ..modules.ema import LitEma
from einops import repeat
from ..util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
)
from einops import rearrange
class DiffusionEngine3D(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        first_stage_config_2d: Union[None, Dict, ListConfig, OmegaConf] = None,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        vae_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        freeze_type: str="none",
        lr_rate: float = 1.0,
        wrapper_type="OPENAIUNETWRAPPERCONTROLLDM3D",
        share_noise_level=0.0,
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.share_noise_level = torch.tensor(share_noise_level)
        self.conditioner_config = conditioner_config
        self.first_stage_config_2d = first_stage_config_2d
        self.network_config = network_config
        self.freeze_type = freeze_type
        self.lr_rate =lr_rate
        self.alpha = network_config.params.alpha
        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)
        self.wrapper_type = eval(wrapper_type)
        wrapper_type = (
            self.wrapper_type if hasattr(self, "wrapper_type") else OPENAIUNETWRAPPER
        )
        self.model = get_obj_from_str(default(network_wrapper, wrapper_type))(
            model, compile_model=compile_model
        )
        self.num_frames = network_config.params.num_frames
        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        if self.freeze_type == "spatial":
            for name, param in self.model.diffusion_model.named_parameters():
                if not "temporal" in name and not "alpha" in name:
                    param.requires_grad = False

        self.setup_vaeembedder()

    def setup_vaeembedder(self):
        for embedder in self.conditioner.embedders:
            if isinstance(embedder, VAEEmbedder):
                embedder.first_stage_model = (
                    self.first_stage_model
                )  # TODO: should we add .clone()
                embedder.disable_first_stage_autocast = (
                    self.disable_first_stage_autocast
                )
                embedder.scale_factor = self.scale_factor
                embedder.freeze()


    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            z = z.to(self.first_stage_model.post_quant_conv.weight.dtype)
            out = self.first_stage_model.decode(z)
        return out


    @torch.no_grad()
    def encode_first_stage(self, x):
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            z = self.first_stage_model.encode(x) #@should change back
        z = self.scale_factor * z
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        x = rearrange(x, "b t c h w -> (b t) c h w ").contiguous()
        x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    def on_train_start(self, *args, **kwargs):
        # FIXME: learning rate may be wrong when resume from checkpoint
        # self.optimizers().param_groups = self.optimizers().optimizer.param_groups
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_st_optimizer_from_config(self, params_spatial, params_temporal,lr_rate,lr, cfg):
        # return get_obj_from_str(cfg["target"])(
        #     params, lr=lr, **cfg.get("params", dict())
        # )
        return get_obj_from_str(cfg["target"])([
            {"params":params_spatial,"lr": lr * lr_rate},
            {"params": params_temporal, "lr": lr}
        ], **cfg.get("params", dict())
        )

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )


    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        if self.share_noise_level > 0.0:
            concat_feat = cond["concat"].to(self.device)
            concat_feat_seq = repeat(concat_feat[-1],"c h w -> t c h w",t=self.num_frames)
            randn = (
                randn + concat_feat_seq * self.share_noise_level
            )
            
        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples


    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[-2:] # change to the last 2
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]

                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                # expand context
                xc = xc.unsqueeze(1).repeat(1, self.num_frames, 1, 1,1)
                xc = rearrange(xc, "b t c h w  -> (b t) c h w ").contiguous()
                log[embedder.input_key] = xc
        return log


    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)
        if 'cond_img' in batch:
            log['cond_img']=rearrange(batch['cond_img'], "b t c h w -> (b t) c h w ").contiguous()
        if 'original_size_as_tuple' in conditioner_input_keys:
            print('sample for xl model')
            c, uc = self.conditioner.get_unconditional_conditioning(
                batch,
                force_uc_zero_embeddings=ucg_keys
                if len(self.conditioner.embedders) > 0
                else [],
            )
        else:
            #print('sample for 2.1 model')
            batch_uc = batch.copy()
            batch_uc["txt"] = ["" for i in
                               batch["txt"]]  # only for sd 2.1, not for sd-xl!

            c, uc = self.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=[]
            )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        B = x.shape[0]
        x = x.to(self.device)[:N]
        x = rearrange(x, "b t c h w -> (b t) c h w ").contiguous()
        log["inputs"] = x
        if 'source_color' in batch:
            color = batch['source_color'].to(self.device)[:N]
            color = rearrange(color, "b t c h w -> (b t) c h w ").contiguous()
            log["source_color"] = color
        z = self.encode_first_stage(x)
        log["reconstructions"] = self.decode_first_stage(z)
        log.update(self.log_conditionings(batch, N))
        if 'cond_feat' in c:
            log["control"] = c['cond_feat'] * 2.0 - 1.0
        for k in c:
            if isinstance(c[k], torch.Tensor):
                if k == 'concat' or k=='cond_bev_feat':
                    c[k], uc[k] = map(lambda y: y[k][:N * self.num_frames].to(self.device), (c, uc))
                elif k == 'cond_feat':
                    c[k], uc[k] = map(lambda y: y[k][:N * self.num_frames * 4].to(self.device), (c, uc)) ##@!! only surport 4 temporal downsample 
                else:
                    c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))            

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=N * self.num_frames, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            log["samples"] = samples

        return log

