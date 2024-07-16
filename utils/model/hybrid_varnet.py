import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.data import transforms

from unet import Unet
from att_varnet import VarNet



class NormUnet(nn.Module):
    """
    Normalized U-Net model.
    """

    def __init__(
        self,
        chans: int,
        num_pool_layers: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        use_attention: bool = True,
        use_res: bool = False,
    ):
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
            use_attention=use_attention,
            use_res=use_res,
        )

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, c * h * w)

        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # normalize
        x, mean, std = self.norm(x)

        x = self.unet(x)

        # unnormalize
        x = self.unnorm(x, mean, std)

        return x


class HybridVarNet(nn.Module):
    """
    A hybrid variational network model.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        use_attention: bool = True,
        use_res: bool = True,
        settings_imgnet: tuple = (8, 4, 0.0),
    ):
        super().__init__()

        self.varnet = VarNet(
            num_cascades,
            sens_chans,
            sens_pools,
            chans,
            pools,
            mask_center,
            use_attention=use_attention,
            use_res=use_res,
        )

        self.imagenet = NormUnet(
            in_chans=1,
            out_chans=1,
            chans=settings_imgnet[0],
            num_pool_layers=settings_imgnet[1],
            drop_prob=settings_imgnet[2],
            use_attention=use_attention,
            use_res=use_res,
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ):

        img_mid = self.varnet(masked_kspace, mask, num_low_frequencies)  # [B,H,W]

        img_in = img_mid[:, None, :, :]  # [B,1,H,W]
        img_out = self.imagenet(img_in)
        img_out = img_out.squeeze(1)  # [B,H,W]

        height = img_out.shape[-2]
        width = img_out.shape[-1]
        return img_out[..., (height - 384) // 2 : 384 + (height - 384) // 2, (width - 384) // 2 : 384 + (width - 384) // 2]