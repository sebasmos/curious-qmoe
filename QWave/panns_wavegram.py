"""PANNs Wavegram-CNN14 (Kong et al., TASLP 2020; github.com/qiuqiangkong/audioset_tagging_cnn, MIT licence; weights Zenodo 3987831,
CC BY 4.0), reimplemented for T23 as a raw-waveform expert, with a W8A8 variant in the conventions of QWave/effnet_quant.py.

Module names follow the authors' Wavegram_Cnn14 so the released state dict loads with torch.load(weights_only=True) and strict=True.
The authors' class also holds bn0 and conv_block1, which its forward never uses; they are kept here only so the state dict loads strictly.
No downloaded code is imported or run.

Input: a 32 kHz waveform [B, n]. Forward (authors' order, eval mode): pre_conv0 (Conv1d 1->64, k 11, stride 5) + BN + ReLU; three
ConvPreWavBlocks (Conv1d k 3, then k 3 dilation 2, each BN + ReLU, max pool 4); reshape the 128 channels into 4 groups of 32 bins and
transpose to [B, 4, frames, 32] (the Wavegram); pre_block4 (ConvBlock 4->64, avg pool (2, 1)); conv_block2 to conv_block6 of CNN14;
mean over the bin axis; max plus mean over time; embedding = ReLU(fc1(.)), 2048-d.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from QWave.effnet_quant import act_scale_dims, fake_quant_sym, rtn_per_channel
from QWave.panns_cnn14 import ConvBlock

SR = 32000


class ConvPreWavBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, 1, padding=1, bias=False)
        self.conv2 = nn.Conv1d(cout, cout, 3, 1, dilation=2, padding=2, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm1d(cout), nn.BatchNorm1d(cout)

    def forward(self, x, pool=4):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.max_pool1d(x, kernel_size=pool)


class WavegramCnn14(nn.Module):
    def __init__(self, classes_num=527):
        super().__init__()
        self.pre_conv0 = nn.Conv1d(1, 64, 11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(4, 64)
        self.bn0 = nn.BatchNorm2d(64)              # unused by the PANNs authors' forward
        chans = [1, 64, 128, 256, 512, 1024, 2048]
        for i in range(6):                          # conv_block1 unused by the PANNs authors' forward
            setattr(self, f"conv_block{i + 1}", ConvBlock(chans[i], chans[i + 1]))
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num)

    def wavegram(self, wave32):
        a = F.relu(self.pre_bn0(self.pre_conv0(wave32[:, None, :])))
        a = self.pre_block1(a, 4)
        a = self.pre_block2(a, 4)
        a = self.pre_block3(a, 4)
        a = a.reshape(a.shape[0], -1, 32, a.shape[-1]).transpose(2, 3)
        return self.pre_block4(a, (2, 1))

    def pooled(self, wave32):
        """Input of fc1: [B, 2048]."""
        x = self.wavegram(wave32)
        for i in range(2, 6):
            x = getattr(self, f"conv_block{i}")(x)
        x = self.conv_block6(x, (1, 1))
        x = x.mean(dim=3)
        return x.max(dim=2)[0] + x.mean(dim=2)

    def forward(self, wave32):
        return F.relu(self.fc1(self.pooled(wave32)))

    def audioset_probs(self, wave32):
        return torch.sigmoid(self.fc_audioset(self.forward(wave32)))


def load_wavegram(path) -> WavegramCnn14:
    """The released checkpoint (weights_only=True); strict load after dropping keys of modules without parameters in our code
    (spec_augmenter, if present). Returns the model in eval mode."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck
    keep = {k: v for k, v in sd.items() if not k.startswith(("spec_augmenter.", "spectrogram_extractor.", "logmel_extractor."))}
    m = WavegramCnn14()
    m.load_state_dict(keep, strict=True)
    return m.eval()


def dropped_keys(path) -> list:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck
    return [k for k in sd if k.startswith(("spec_augmenter.", "spectrogram_extractor.", "logmel_extractor."))]


class QuantWavegram(nn.Module):
    """W8A8 copy: per-output-channel symmetric RTN weights on every Conv1d, Conv2d and fc1, 8-bit dynamic symmetric inputs
    (Conv1d: one scale per clip over channels and time; Conv2d: effnet_quant.act_scale_dims, per clip for 3x3; fc1: one per row).
    pre_conv0 reads the raw waveform and stays FP32 (it is priced with the front end at 32 x 32, as the mel front ends are).
    BatchNorm, ReLU, pooling in FP32."""

    def __init__(self, base: WavegramCnn14, wbits=8, abits=8, device="cpu"):
        super().__init__()
        m = copy.deepcopy(base).cpu().eval()
        self.abits = abits
        with torch.no_grad():
            for name, mod in m.named_modules():
                if mod is m.pre_conv0 or mod is m.fc_audioset:
                    continue
                if isinstance(mod, nn.Conv1d):
                    dims = (1, 2)
                elif isinstance(mod, nn.Conv2d):
                    dims = act_scale_dims(mod)
                elif isinstance(mod, nn.Linear):
                    dims = (-1,)
                else:
                    continue
                codes, s = rtn_per_channel(mod.weight.data, wbits)
                mod.weight.data.copy_(codes * s)
                mod.register_forward_pre_hook(self._hook(dims))
        self.net = m.to(device)

    def _hook(self, dims):
        def hook(mod, inp):
            return (fake_quant_sym(inp[0], self.abits, dims),) + tuple(inp[1:])
        return hook

    @torch.no_grad()
    def forward(self, x):
        return self.net(x)


def macs_at(model: WavegramCnn14, n32: int) -> dict:
    """MACs and output elements for one clip of n32 samples at 32 kHz, by hooks on a zero input over the modules the forward uses:
    'front' = pre_conv0 (FP32), 'macs'/'out_elems' = every other Conv1d, Conv2d and fc1 (the W8A8 part)."""
    tot = {"front": 0.0, "macs": 0.0, "out_elems": 0.0}

    def hook(mod, inp, out):
        if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            k = 1
            for v in mod.kernel_size:
                k *= v
            n_out = out[0].numel()
            macs = n_out * (mod.in_channels // mod.groups) * k
        else:
            n_out = mod.out_features
            macs = mod.in_features * mod.out_features
        if mod is model.pre_conv0:
            tot["front"] += macs
        else:
            tot["macs"] += macs
            tot["out_elems"] += n_out

    used = [model.pre_conv0, model.fc1] + [mm for nm, mm in model.named_modules()
                                           if isinstance(mm, (nn.Conv1d, nn.Conv2d)) and mm is not model.pre_conv0
                                           and not nm.startswith("conv_block1.")]
    hs = [mm.register_forward_hook(hook) for mm in used]
    try:
        with torch.no_grad():
            model.eval()(torch.zeros(1, n32))
    finally:
        for h in hs:
            h.remove()
    return tot
