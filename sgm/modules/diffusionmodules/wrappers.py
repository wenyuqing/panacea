import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"
OPENAIUNETWRAPPERCONTROLLDM3D = (
    "sgm.modules.diffusionmodules.wrappers.OpenAIWrapperControlLDM3D"
)

class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs,
        )
class OpenAIWrapperControlLDM3D(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        cond_feat = c["cond_feat"]

        model_dtype = self.diffusion_model.controlnet.input_hint_block[0].weight.dtype
        x = x.to(model_dtype)
        cond_feat = cond_feat.to(model_dtype)

        c['crossattn']=c['crossattn'].to(model_dtype)
        control = self.diffusion_model.controlnet(
            x=x,  # noisy control image, use or not used it depend on control_model style
            hint=cond_feat,  # control image B C H W
            timesteps=t,  # time step
            context=c.get(
                "crossattn", None
            ),  # text prompt, use or not used it depend on control_model style
            y=c.get("vector", None),
            **kwargs
        )

        out = self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            control=control,
            only_mid_control=False,
            **kwargs
        )

        return out