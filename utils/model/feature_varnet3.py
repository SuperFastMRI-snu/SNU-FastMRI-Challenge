"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import NamedTuple, Optional, Tuple, List
import math
import torch
import torch.nn as nn
from torch import Tensor

import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import math
from fastmri.data.transforms import center_crop, batched_mask_center
from fastmri.fftc import ifft2c_new as ifft2c
from fastmri.fftc import fft2c_new as fft2c
from fastmri.coil_combine import rss_complex, rss
from fastmri.math import complex_abs, complex_mul, complex_conj

from unet import Unet, ConvBlock, TransposeConvBlock
from varnet import NormUnet, SensitivityModel, VarNetBlock

def image_crop(image: Tensor, crop_size: Optional[Tuple[int, int]] = None) -> Tensor:
    if crop_size is None:
        return image
    return center_crop(image, crop_size).contiguous()


def _calc_uncrop(crop_height: int, in_height: int) -> Tuple[int, int]:
    pad_height = (in_height - crop_height) // 2
    if (in_height - crop_height) % 2 != 0:
        pad_height_top = pad_height + 1
    else:
        pad_height_top = pad_height

    pad_height = in_height - pad_height

    return pad_height_top, pad_height


def image_uncrop(image: Tensor, original_image: Tensor) -> Tensor:
    """Insert values back into original image."""
    in_shape = original_image.shape
    original_image = original_image.clone()

    if in_shape == image.shape:
        return image

    pad_height_top, pad_height = _calc_uncrop(image.shape[-2], in_shape[-2])
    pad_height_left, pad_width = _calc_uncrop(image.shape[-1], in_shape[-1])

    try:
        if len(in_shape) == 2:  # Assuming 2D images
            original_image[pad_height_top:pad_height, pad_height_left:pad_width] = image
        elif len(in_shape) == 3:  # Assuming 3D images with channels
            original_image[
                :, pad_height_top:pad_height, pad_height_left:pad_width
            ] = image
        elif len(in_shape) == 4:  # Assuming 4D images with batch size
            original_image[
                :, :, pad_height_top:pad_height, pad_height_left:pad_width
            ] = image
        else:
            raise RuntimeError(f"Unsupported tensor shape: {in_shape}")
    except RuntimeError:
        print(f"in_shape: {in_shape}, image shape: {image.shape}")
        raise

    return original_image


def norm_fn(image: Tensor, means: Tensor, variances: Tensor) -> Tensor:
    means = means.view(1, -1, 1, 1)
    variances = variances.view(1, -1, 1, 1)
    return (image - means) * torch.rsqrt(variances)


def unnorm_fn(image: Tensor, means: Tensor, variances: Tensor) -> Tensor:
    means = means.view(1, -1, 1, 1)
    variances = variances.view(1, -1, 1, 1)
    return image * torch.sqrt(variances) + means


def complex_to_chan_dim(x: Tensor) -> Tensor:
    b, c, h, w, two = x.shape
    assert two == 2
    assert c == 1
    return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)


def chan_complex_to_last_dim(x: Tensor) -> Tensor:
    b, c2, h, w = x.shape
    assert c2 == 2
    c = c2 // 2
    return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()


def sens_expand(x: Tensor, sens_maps: Tensor) -> Tensor:
    return fft2c(complex_mul(chan_complex_to_last_dim(x), sens_maps))


def sens_reduce(x: Tensor, sens_maps: Tensor) -> Tensor:
    return complex_to_chan_dim(
        complex_mul(ifft2c(x), complex_conj(sens_maps)).sum(dim=1, keepdim=True)
    )


class NormStats(nn.Module):
    def forward(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        # group norm
        batch, chans, _, _ = data.shape

        if batch != 1:
            raise ValueError("Unexpected input dimensions.")

        data = data.view(chans, -1)

        mean = data.mean(dim=1)
        variance = data.var(dim=1, unbiased=False)

        assert mean.ndim == 1
        assert variance.ndim == 1
        assert mean.shape[0] == chans
        assert variance.shape[0] == chans

        return mean, variance

class FeatureImage(NamedTuple):
    features: Tensor
    acceleration: Optional[int] = None
    sens_maps: Optional[Tensor] = None
    crop_size: Optional[Tuple[int, int]] = None
    means: Optional[Tensor] = None
    variances: Optional[Tensor] = None
    mask: Optional[Tensor] = None
    ref_kspace: Optional[Tensor] = None
    beta: Optional[Tensor] = None
    gamma: Optional[Tensor] = None


class FeatureEncoder(nn.Module):
    def __init__(self, in_chans: int, feature_chans: int = 32, drop_prob: float = 0.0):
        super().__init__()
        self.feature_chans = feature_chans

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=feature_chans,
                kernel_size=5,
                padding=2,
                bias=True,
            ),
        )

    def forward(self, image: Tensor, means: Tensor, variances: Tensor) -> Tensor:
        means = means.view(1, -1, 1, 1)
        variances = variances.view(1, -1, 1, 1)
        return self.encoder((image - means) * torch.rsqrt(variances))


class FeatureDecoder(nn.Module):
    def __init__(self, feature_chans: int = 32, out_chans: int = 2):
        super().__init__()
        self.feature_chans = feature_chans

        self.decoder = nn.Conv2d(
            in_channels=feature_chans,
            out_channels=out_chans,
            kernel_size=5,
            padding=2,
            bias=True,
        )

    def forward(self, features: Tensor, means: Tensor, variances: Tensor) -> Tensor:
        means = means.view(1, -1, 1, 1)
        variances = variances.view(1, -1, 1, 1)
        return self.decoder(features) * torch.sqrt(variances) + means


class AttentionPE(nn.Module):
    def __init__(self, in_chans: int):
        super().__init__()
        self.in_chans = in_chans

        self.norm = nn.InstanceNorm2d(in_chans)
        self.q = nn.Conv2d(in_chans, in_chans, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_chans, in_chans, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_chans, in_chans, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            in_chans, in_chans, kernel_size=1, stride=1, padding=0
        )
        self.dilated_conv = nn.Conv2d(
            in_chans, in_chans, kernel_size=3, stride=1, padding=2, dilation=2
        )

    def reshape_to_blocks(self, x: Tensor, accel: int) -> Tensor:
        chans = x.shape[1]
        pad_total = (accel - (x.shape[3] - accel)) % accel
        pad_right = pad_total // 2
        pad_left = pad_total - pad_right
        x = F.pad(x, (pad_left, pad_right, 0, 0), "reflect")
        return (
            torch.stack(x.chunk(chunks=accel, dim=3), dim=-1)
            .view(chans, -1, accel)
            .permute(1, 0, 2)
            .contiguous()
        )

    def reshape_from_blocks(
        self, x: Tensor, image_size: Tuple[int, int], accel: int
    ) -> Tensor:
        chans = x.shape[1]
        num_freq, num_phase = image_size
        x = (
            x.permute(1, 0, 2)
            .reshape(1, chans, num_freq, -1, accel)
            .permute(0, 1, 2, 4, 3)
            .reshape(1, chans, num_freq, -1)
        )
        padded_phase = x.shape[3]
        pad_total = padded_phase - num_phase
        pad_right = pad_total // 2
        pad_left = pad_total - pad_right
        return x[:, :, :, pad_left : padded_phase - pad_right]

    def get_positional_encodings(
        self, seq_len: int, embed_dim: int, device: str
    ) -> Tensor:
        freqs = torch.tensor(
            [1 / (10000 ** (2 * (i // 2) / embed_dim)) for i in range(embed_dim)],
            device=device,
        )
        freqs = freqs.unsqueeze(0)
        positions = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        scaled = positions * freqs
        sin_encodings = torch.sin(scaled)
        cos_encodings = torch.cos(scaled)
        encodings = torch.cat([sin_encodings, cos_encodings], dim=1)[:, :embed_dim]
        return encodings

    def forward(self, x: Tensor, accel: int) -> Tensor:
        im_size = (x.shape[2], x.shape[3])
        h_ = x
        h_ = self.norm(h_)

        pos_enc = self.get_positional_encodings(x.shape[2], x.shape[3], h_.device.type)

        h_ = h_ + pos_enc

        q = self.dilated_conv(self.q(h_))
        k = self.dilated_conv(self.k(h_))
        v = self.dilated_conv(self.v(h_))

        # compute attention
        c = q.shape[1]
        q = self.reshape_to_blocks(q, accel)
        k = self.reshape_to_blocks(k, accel)
        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = self.reshape_to_blocks(v, accel)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = self.reshape_from_blocks(h_, im_size, accel)

        h_ = self.proj_out(h_)

        return x + h_

class Unet2d(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        output_bias: bool = False,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_planes = out_chans
        self.factor = 2**num_pool_layers

        # Build from the middle of the UNet outwards
        planes = 2 ** (num_pool_layers)
        layer = None
        for _ in range(num_pool_layers):
            planes = planes // 2
            layer = UnetLevel(
                layer,
                in_planes=planes * chans,
                out_planes=2 * planes * chans,
                drop_prob=drop_prob,
            )

        self.layer = UnetLevel(
            layer, in_planes=in_chans, out_planes=chans, drop_prob=drop_prob
        )

        if output_bias:
            self.final_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=chans,
                    out_channels=out_chans,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            )
        else:
            self.final_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=chans,
                    out_channels=out_chans,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

    def pad_input_image(self, image: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        # pad image if it's not divisible by downsamples
        _, _, height, width = image.shape
        pad_height = (self.factor - (height - self.factor)) % self.factor
        pad_width = (self.factor - (width - self.factor)) % self.factor
        if pad_height != 0 or pad_width != 0:
            image = F.pad(image, (0, pad_width, 0, pad_height), mode="reflect")

        return image, (height, width)

    def forward(self, image: Tensor) -> Tensor:
        image, (output_y, output_x) = self.pad_input_image(image)
        return self.final_conv(self.layer(image))[:, :, :output_y, :output_x]


class UnetLevel(nn.Module):
    def __init__(
        self,
        child: Optional[nn.Module],
        in_planes: int,
        out_planes: int,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.left_block = ConvBlock(
            in_chans=in_planes, out_chans=out_planes, drop_prob=drop_prob
        )

        self.child = child

        if child is not None:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            if isinstance(child, UnetLevel):  # Ensure child is an instance of UnetLevel
                self.upsample = TransposeConvBlock(
                    in_chans=child.out_planes, out_chans=out_planes
                )
            else:
                raise TypeError("Child must be an instance of UnetLevel")

            self.right_block = ConvBlock(
                in_chans=2 * out_planes, out_chans=out_planes, drop_prob=drop_prob
            )

    def down_up(self, image: Tensor) -> Tensor:
        if self.child is None:
            raise ValueError("self.child is None, cannot call down_up.")
        downsampled = self.downsample(image)
        child_output = self.child(downsampled)
        upsampled = self.upsample(child_output)
        return upsampled

    def forward(self, image: Tensor) -> Tensor:
        image = self.left_block(image)

        if self.child is not None:
            image = self.right_block(torch.cat((image, self.down_up(image)), 1))

        return image


class Norm1DUnet(nn.Module):
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):

        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)

        mean = x.mean()
        std = x.std()

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        # attention_goes_here
        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class FIVarNet_acc_fit(nn.Module):
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        image_conv_cascades: Optional[List[int]] = None,
        kspace_mult_factor: float = 1e6,
    ):
        super().__init__()
        if image_conv_cascades is None:
            image_conv_cascades = [ind for ind in range(num_cascades) if ind % 3 == 0]

        self.image_conv_cascades = image_conv_cascades
        self.kspace_mult_factor = kspace_mult_factor
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.encoder = FeatureEncoder(in_chans=2, feature_chans=chans)
        self.decoder = FeatureDecoder(feature_chans=chans, out_chans=2)
        cascades = []
        for ind in range(num_cascades):
            use_image_conv = ind in self.image_conv_cascades
            cascades.append(
                AttentionFeatureVarNetBlock(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    feature_processor=Unet2d(
                        in_chans=chans, out_chans=chans, chans=chans, num_pool_layers=pools
                    ),
                    attention_layer=AttentionPE(in_chans=chans),
                    use_extra_feature_conv=use_image_conv,
                )
            )

        self.image_cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )

        self.decode_norm = nn.InstanceNorm2d(chans)
        self.cascades = nn.Sequential(*cascades)
        self.norm_fn = NormStats()

    def _decode_output(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(
            self.decode_norm(feature_image.features),
            means=feature_image.means,
            variances=feature_image.variances,
        )
        return sens_expand(image, feature_image.sens_maps)

    def _encode_input(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        acceleration: int,
        crop_size: Optional[Tuple[int, int]],
        num_low_frequencies: Optional[int],
    ) -> FeatureImage:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        image = sens_reduce(masked_kspace, sens_maps)
        # detect FLAIR 203
        if crop_size is not None and image.shape[-1] < crop_size[1]:
            crop_size = (image.shape[-1], image.shape[-1])
        means, variances = self.norm_fn(image)
        features = self.encoder(image, means=means, variances=variances)

        return FeatureImage(
            features=features,
            acceleration=acceleration,
            sens_maps=sens_maps,
            crop_size=crop_size,
            means=means,
            variances=variances,
            ref_kspace=masked_kspace,
            mask=mask,
        )

    def forward(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        acceleration: int,
        num_low_frequencies: Optional[int] = None,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        masked_kspace = masked_kspace * self.kspace_mult_factor
        # Encode to features and get sensitivities
        feature_image = self._encode_input(
            masked_kspace=masked_kspace,
            mask=mask,
            acceleration=acceleration,
            crop_size=crop_size,
            num_low_frequencies=num_low_frequencies,
        )
        # Do DC in feature-space
        feature_image = self.cascades(feature_image)
        # Find last k-space
        kspace_pred = self._decode_output(feature_image)
        # Run E2EVN
        for cascade in self.image_cascades:
            kspace_pred = cascade(
                kspace_pred, feature_image.ref_kspace, mask, feature_image.sens_maps
            )
        # Divide with k-space factor and Return Final Image
        kspace_pred = (
            kspace_pred / self.kspace_mult_factor
        )  # Ensure kspace_pred is a Tensor
        result = rss(
            complex_abs(ifft2c(kspace_pred)), dim=1
        )  # Ensure kspace_pred is a Tensor
        height = result.shape[-2]
        width = result.shape[-1]
        return result[..., (height - 384) // 2 : 384 + (height - 384) // 2, (width - 384) // 2 : 384 + (width - 384) // 2]


class IFVarNet(nn.Module):
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        image_conv_cascades: Optional[List[int]] = None,
        kspace_mult_factor: float = 1e6,
    ):
        super().__init__()
        if image_conv_cascades is None:
            image_conv_cascades = [ind for ind in range(num_cascades) if ind % 3 == 0]

        self.image_conv_cascades = image_conv_cascades
        self.kspace_mult_factor = kspace_mult_factor
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.encoder = FeatureEncoder(in_chans=2, feature_chans=chans)
        self.decoder = FeatureDecoder(feature_chans=chans, out_chans=2)
        cascades = []
        for ind in range(num_cascades):
            use_image_conv = ind in self.image_conv_cascades
            cascades.append(
                AttentionFeatureVarNetBlock(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    feature_processor=Unet2d(
                        in_chans=chans, out_chans=chans, chans=chans, num_pool_layers=pools
                    ),
                    attention_layer=AttentionPE(in_chans=chans),
                    use_extra_feature_conv=use_image_conv,
                )
            )

        self.image_cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )

        self.decode_norm = nn.InstanceNorm2d(chans)
        self.cascades = nn.Sequential(*cascades)
        self.norm_fn = NormStats()

    def _decode_output(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(
            self.decode_norm(feature_image.features),
            means=feature_image.means,
            variances=feature_image.variances,
        )
        return sens_expand(image, feature_image.sens_maps)

    def _encode_input(
        self,
        masked_kspace: Tensor,
        ref_kspace: Tensor,
        sens_maps: Tensor,
        mask: Tensor,
        crop_size: Optional[Tuple[int, int]],
    ) -> FeatureImage:
        image = sens_reduce(masked_kspace, sens_maps)
        # detect FLAIR 203
        if crop_size is not None and image.shape[-1] < crop_size[1]:
            crop_size = (image.shape[-1], image.shape[-1])
        means, variances = self.norm_fn(image)
        features = self.encoder(image, means=means, variances=variances)

        return FeatureImage(
            features=features,
            sens_maps=sens_maps,
            crop_size=crop_size,
            means=means,
            variances=variances,
            ref_kspace=ref_kspace,
            mask=mask,
        )

    def forward(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        num_low_frequencies: Optional[int] = None,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:

        masked_kspace = masked_kspace * self.kspace_mult_factor

        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        kspace_pred = masked_kspace.clone()
        # Run E2EVN
        for cascade in self.image_cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        feature_image = self._encode_input(
            masked_kspace=kspace_pred,
            ref_kspace=masked_kspace,
            sens_maps=sens_maps,
            mask=mask,
            crop_size=crop_size,
        )
        feature_image = self.cascades(feature_image)
        kspace_pred = self._decode_output(feature_image)
        kspace_pred = (
            kspace_pred / self.kspace_mult_factor
        )  # Ensure kspace_pred is a Tensor
        return rss(
            complex_abs(ifft2c(kspace_pred)), dim=1
        )  # Ensure kspace_pred is a Tensor


class FeatureVarNet_sh_w(nn.Module):
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        image_conv_cascades: Optional[List[int]] = None,
        kspace_mult_factor: float = 1e6,
    ):
        super().__init__()
        if image_conv_cascades is None:
            image_conv_cascades = [ind for ind in range(num_cascades) if ind % 3 == 0]

        self.image_conv_cascades = image_conv_cascades
        self.kspace_mult_factor = kspace_mult_factor
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.encoder = FeatureEncoder(in_chans=2, feature_chans=chans)
        self.decoder = FeatureDecoder(feature_chans=chans, out_chans=2)
        cascades = []
        for ind in range(num_cascades):
            use_image_conv = ind in self.image_conv_cascades
            cascades.append(
                FeatureVarNetBlock(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    feature_processor=Unet2d(
                        in_chans=chans, out_chans=chans, chans=chans, num_pool_layers=pools
                    ),
                    use_extra_feature_conv=use_image_conv,
                )
            )

        self.decode_norm = nn.InstanceNorm2d(chans)
        self.cascades = nn.Sequential(*cascades)
        self.norm_fn = NormStats()

    def _decode_output(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(
            self.decode_norm(feature_image.features),
            means=feature_image.means,
            variances=feature_image.variances,
        )
        return sens_expand(image, feature_image.sens_maps)

    def _encode_input(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        crop_size: Optional[Tuple[int, int]],
        num_low_frequencies: Optional[int],
    ) -> FeatureImage:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        image = sens_reduce(masked_kspace, sens_maps)
        # detect FLAIR 203
        if crop_size is not None and image.shape[-1] < crop_size[1]:
            crop_size = (image.shape[-1], image.shape[-1])
        means, variances = self.norm_fn(image)
        features = self.encoder(image, means=means, variances=variances)

        return FeatureImage(
            features=features,
            sens_maps=sens_maps,
            crop_size=crop_size,
            means=means,
            variances=variances,
            ref_kspace=masked_kspace,
            mask=mask,
        )

    def forward(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        num_low_frequencies: Optional[int] = None,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        masked_kspace = masked_kspace * self.kspace_mult_factor
        # Encode to features and get sensitivities
        feature_image = self._encode_input(
            masked_kspace=masked_kspace,
            mask=mask,
            crop_size=crop_size,
            num_low_frequencies=num_low_frequencies,
        )
        # Do DC in feature-space
        feature_image = self.cascades(feature_image)
        # Find last k-space
        kspace_pred = self._decode_output(feature_image)
        # Return Final Image
        kspace_pred = (
            kspace_pred / self.kspace_mult_factor
        )  # Ensure kspace_pred is a Tensor
        return rss(
            complex_abs(ifft2c(kspace_pred)), dim=1
        )  # Ensure kspace_pred is a Tensor


class FeatureVarNet_n_sh_w(nn.Module):
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        image_conv_cascades: Optional[List[int]] = None,
        kspace_mult_factor: float = 1e6,
    ):
        super().__init__()
        if image_conv_cascades is None:
            image_conv_cascades = [ind for ind in range(num_cascades) if ind % 3 == 0]

        self.image_conv_cascades = image_conv_cascades
        self.kspace_mult_factor = kspace_mult_factor
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.encoder = FeatureEncoder(in_chans=2, feature_chans=chans)
        self.decoder = FeatureDecoder(feature_chans=chans, out_chans=2)
        cascades = []
        for ind in range(num_cascades):
            use_image_conv = ind in self.image_conv_cascades
            cascades.append(
                FeatureVarNetBlock(
                    encoder=FeatureEncoder(in_chans=2, feature_chans=chans),
                    decoder=FeatureDecoder(feature_chans=chans, out_chans=2),
                    feature_processor=Unet2d(
                        in_chans=chans, out_chans=chans, chans=chans, num_pool_layers=pools
                    ),
                    use_extra_feature_conv=use_image_conv,
                )
            )

        self.decode_norm = nn.InstanceNorm2d(chans)
        self.cascades = nn.Sequential(*cascades)
        self.norm_fn = NormStats()

    def _decode_output(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(
            self.decode_norm(feature_image.features),
            means=feature_image.means,
            variances=feature_image.variances,
        )
        return sens_expand(image, feature_image.sens_maps)

    def _encode_input(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        crop_size: Optional[Tuple[int, int]],
        num_low_frequencies: Optional[int],
    ) -> FeatureImage:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        image = sens_reduce(masked_kspace, sens_maps)
        # detect FLAIR 203
        if crop_size is not None and image.shape[-1] < crop_size[1]:
            crop_size = (image.shape[-1], image.shape[-1])
        means, variances = self.norm_fn(image)
        features = self.encoder(image, means=means, variances=variances)

        return FeatureImage(
            features=features,
            sens_maps=sens_maps,
            crop_size=crop_size,
            means=means,
            variances=variances,
            ref_kspace=masked_kspace,
            mask=mask,
        )

    def forward(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        num_low_frequencies: Optional[int] = None,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        masked_kspace = masked_kspace * self.kspace_mult_factor
        # Encode to features and get sensitivities
        feature_image = self._encode_input(
            masked_kspace=masked_kspace,
            mask=mask,
            crop_size=crop_size,
            num_low_frequencies=num_low_frequencies,
        )
        # Do DC in feature-space
        feature_image = self.cascades(feature_image)
        # Find last k-space
        kspace_pred = self._decode_output(feature_image)
        # Return Final Image
        kspace_pred = (
            kspace_pred / self.kspace_mult_factor
        )  # Ensure kspace_pred is a Tensor
        return rss(
            complex_abs(ifft2c(kspace_pred)), dim=1
        )  # Ensure kspace_pred is a Tensor


class AttentionFeatureVarNet_n_sh_w(nn.Module):
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        image_conv_cascades: Optional[List[int]] = None,
        kspace_mult_factor: float = 1e6,
    ):
        super().__init__()
        if image_conv_cascades is None:
            image_conv_cascades = [ind for ind in range(num_cascades) if ind % 3 == 0]

        self.image_conv_cascades = image_conv_cascades
        self.kspace_mult_factor = kspace_mult_factor
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.encoder = FeatureEncoder(in_chans=2, feature_chans=chans)
        self.decoder = FeatureDecoder(feature_chans=chans, out_chans=2)
        cascades = []
        for ind in range(num_cascades):
            use_image_conv = ind in self.image_conv_cascades
            cascades.append(
                AttentionFeatureVarNetBlock(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    feature_processor=Unet2d(
                        in_chans=chans, out_chans=chans, chans=chans, num_pool_layers=pools
                    ),
                    attention_layer=AttentionPE(in_chans=chans),
                    use_extra_feature_conv=use_image_conv,
                )
            )

        self.decode_norm = nn.InstanceNorm2d(chans)
        self.cascades = nn.Sequential(*cascades)
        self.norm_fn = NormStats()

    def _decode_output(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(
            self.decode_norm(feature_image.features),
            means=feature_image.means,
            variances=feature_image.variances,
        )
        return sens_expand(image, feature_image.sens_maps)

    def _encode_input(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        crop_size: Optional[Tuple[int, int]],
        num_low_frequencies: Optional[int],
    ) -> FeatureImage:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        image = sens_reduce(masked_kspace, sens_maps)
        # detect FLAIR 203
        if crop_size is not None and image.shape[-1] < crop_size[1]:
            crop_size = (image.shape[-1], image.shape[-1])
        means, variances = self.norm_fn(image)
        features = self.encoder(image, means=means, variances=variances)

        return FeatureImage(
            features=features,
            sens_maps=sens_maps,
            crop_size=crop_size,
            means=means,
            variances=variances,
            ref_kspace=masked_kspace,
            mask=mask,
        )

    def forward(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        num_low_frequencies: Optional[int] = None,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        masked_kspace = masked_kspace * self.kspace_mult_factor
        # Encode to features and get sensitivities
        feature_image = self._encode_input(
            masked_kspace=masked_kspace,
            mask=mask,
            crop_size=crop_size,
            num_low_frequencies=num_low_frequencies,
        )
        # Do DC in feature-space
        feature_image = self.cascades(feature_image)
        # Find last k-space
        kspace_pred = self._decode_output(feature_image)
        # Return Final Image
        kspace_pred = (
            kspace_pred / self.kspace_mult_factor
        )  # Ensure kspace_pred is a Tensor
        return rss(
            complex_abs(ifft2c(kspace_pred)), dim=1
        )  # Ensure kspace_pred is a Tensor


class AttentionFeatureVarNetBlock(nn.Module):
    def __init__(
        self,
        encoder: FeatureEncoder,
        decoder: FeatureDecoder,
        feature_processor: Unet2d,
        attention_layer: AttentionPE,
        use_extra_feature_conv: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.feature_processor = feature_processor
        self.attention_layer = attention_layer
        self.use_image_conv = use_extra_feature_conv
        self.dc_weight = nn.Parameter(torch.ones(1))
        feature_chans = self.encoder.feature_chans

        self.input_norm = nn.InstanceNorm2d(feature_chans)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if use_extra_feature_conv:
            self.output_norm = nn.InstanceNorm2d(feature_chans)
            self.output_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=feature_chans,
                    out_channels=feature_chans,
                    kernel_size=5,
                    padding=2,
                    bias=False,
                ),
                nn.InstanceNorm2d(feature_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(
                    in_channels=feature_chans,
                    out_channels=feature_chans,
                    kernel_size=5,
                    padding=2,
                    bias=False,
                ),
                nn.InstanceNorm2d(feature_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

        self.zero: Tensor
        self.register_buffer("zero", torch.zeros(1, 1, 1, 1, 1))

    def encode_from_kspace(self, kspace: Tensor, feature_image: FeatureImage) -> Tensor:
        image = sens_reduce(kspace, feature_image.sens_maps)

        return self.encoder(
            image, means=feature_image.means, variances=feature_image.variances
        )

    def decode_to_kspace(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(
            feature_image.features,
            means=feature_image.means,
            variances=feature_image.variances,
        )

        return sens_expand(image, feature_image.sens_maps)

    def compute_dc_term(self, feature_image: FeatureImage) -> Tensor:
        est_kspace = self.decode_to_kspace(feature_image)

        return self.dc_weight * self.encode_from_kspace(
            torch.where(
                feature_image.mask, est_kspace - feature_image.ref_kspace, self.zero
            ),
            feature_image,
        )

    def apply_model_with_crop(self, feature_image: FeatureImage) -> Tensor:
        if feature_image.crop_size is not None:
            features = image_uncrop(
                self.feature_processor(
                    image_crop(feature_image.features, feature_image.crop_size)
                ),
                feature_image.features.clone(),
            )
        else:
            features = self.feature_processor(feature_image.features)

        return features

    def forward(self, feature_image: FeatureImage, acceleration: Optional [int] = 4) -> FeatureImage:
        feature_image = feature_image._replace(
            features=self.input_norm(feature_image.features)
        )

        new_features = feature_image.features - self.compute_dc_term(feature_image)
        """
        new_features_np = feature_image.features.cpu().numpy()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = f'new_features_before_{timestamp}.mat'
        savemat(file_name, {'new_features_before': new_features_np})

        new_ref_kspace = feature_image.ref_kspace.cpu().numpy()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = f'kspace_{timestamp}.mat'
        savemat(file_name, {'kspace_': new_ref_kspace})
        """
        feature_image = feature_image._replace(
            features=self.attention_layer(feature_image.features, feature_image.acceleration)
        )
        new_features = new_features - self.apply_model_with_crop(feature_image)

        if self.use_image_conv:
            new_features = self.output_norm(new_features)
            new_features = new_features + self.output_conv(new_features)

        return feature_image._replace(features=new_features)


class FeatureVarNetBlock(nn.Module):
    def __init__(
        self,
        encoder: FeatureEncoder,
        decoder: FeatureDecoder,
        feature_processor: Unet2d,
        use_extra_feature_conv: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.feature_processor = feature_processor
        self.use_image_conv = use_extra_feature_conv
        self.dc_weight = nn.Parameter(torch.ones(1))
        feature_chans = self.encoder.feature_chans

        self.input_norm = nn.InstanceNorm2d(feature_chans)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if use_extra_feature_conv:
            self.output_norm = nn.InstanceNorm2d(feature_chans)
            self.output_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=feature_chans,
                    out_channels=feature_chans,
                    kernel_size=5,
                    padding=2,
                    bias=False,
                ),
                nn.InstanceNorm2d(feature_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(
                    in_channels=feature_chans,
                    out_channels=feature_chans,
                    kernel_size=5,
                    padding=2,
                    bias=False,
                ),
                nn.InstanceNorm2d(feature_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

        self.zero: Tensor
        self.register_buffer("zero", torch.zeros(1, 1, 1, 1, 1))

    def encode_from_kspace(self, kspace: Tensor, feature_image: FeatureImage) -> Tensor:
        image = sens_reduce(kspace, feature_image.sens_maps)

        return self.encoder(
            image, means=feature_image.means, variances=feature_image.variances
        )

    def decode_to_kspace(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(
            feature_image.features,
            means=feature_image.means,
            variances=feature_image.variances,
        )

        return sens_expand(image, feature_image.sens_maps)

    def compute_dc_term(self, feature_image: FeatureImage) -> Tensor:
        est_kspace = self.decode_to_kspace(feature_image)

        return self.dc_weight * self.encode_from_kspace(
            torch.where(
                feature_image.mask, est_kspace - feature_image.ref_kspace, self.zero
            ),
            feature_image,
        )

    def apply_model_with_crop(self, feature_image: FeatureImage) -> Tensor:
        if feature_image.crop_size is not None:
            features = image_uncrop(
                self.feature_processor(
                    image_crop(feature_image.features, feature_image.crop_size)
                ),
                feature_image.features.clone(),
            )
        else:
            features = self.feature_processor(feature_image.features)

        return features

    def forward(self, feature_image: FeatureImage) -> FeatureImage:
        feature_image = feature_image._replace(
            features=self.input_norm(feature_image.features)
        )

        new_features = (
            feature_image.features
            - self.compute_dc_term(feature_image)
            - self.apply_model_with_crop(feature_image)
        )

        if self.use_image_conv:
            new_features = self.output_norm(new_features)
            new_features = new_features + self.output_conv(new_features)

        return feature_image._replace(features=new_features)