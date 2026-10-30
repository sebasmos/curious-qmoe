"""T23 fallback: the deployed W8A8 tail (conv_block6 + fc1 of a fine-tuned Wavegram-CNN14) stored compactly. The three quantized weights
(conv_block6.conv1, conv_block6.conv2, fc1) are kept as int8 RTN codes plus FP32 per-output-channel scales (QWave.effnet_quant.rtn_per_channel,
the function QuantWavegram uses), every other tail tensor in FP32. Loading rebuilds codes * scales, the weights QuantWavegram builds, and the
same dynamic 8-bit activation hooks, so TailQ8 on the W8A8 trunk output equals QuantWavegram's conv_block6 + fc1 (checked in t23_ft_score.py)."""
import copy
import torch
import torch.nn.functional as F
from torch import nn
from QWave.effnet_quant import act_scale_dims, fake_quant_sym, rtn_per_channel

QW = ("conv_block6.conv1.weight", "conv_block6.conv2.weight", "fc1.weight")


def save_tail_q8(conv_block6, fc1, path):
    sd = {}
    for pre, mod in (("conv_block6.", conv_block6), ("fc1.", fc1)):
        for k, v in mod.state_dict().items():
            if pre + k in QW:
                codes, s = rtn_per_channel(v.detach().float().cpu(), 8)
                sd[pre + k + ".codes"], sd[pre + k + ".scale"] = codes.to(torch.int8), s.contiguous()
            else:
                sd[pre + k] = v.detach().cpu().clone()
    torch.save(sd, path)


class TailQ8(nn.Module):
    def __init__(self, base, path, device):
        super().__init__()
        self.conv_block6, self.fc1 = copy.deepcopy(base.conv_block6).cpu(), copy.deepcopy(base.fc1).cpu()
        sd = torch.load(path, map_location="cpu", weights_only=True); full = {}
        for k, v in sd.items():
            if k.endswith(".codes"): full[k[:-6]] = v.float() * sd[k[:-6] + ".scale"]
            elif not k.endswith(".scale"): full[k] = v
        self.conv_block6.load_state_dict({k[12:]: v for k, v in full.items() if k.startswith("conv_block6.")}, strict=True)
        self.fc1.load_state_dict({k[4:]: v for k, v in full.items() if k.startswith("fc1.")}, strict=True)
        for mod, dims in ((self.conv_block6.conv1, act_scale_dims(self.conv_block6.conv1)), (self.conv_block6.conv2, act_scale_dims(self.conv_block6.conv2)), (self.fc1, (-1,))):
            mod.register_forward_pre_hook(lambda m_, inp, d=dims: (fake_quant_sym(inp[0], 8, d),) + tuple(inp[1:]))
        self.to(device).eval()

    @torch.no_grad()
    def forward(self, x):
        x = self.conv_block6(x, (1, 1)); x = x.mean(3); x = x.max(2)[0] + x.mean(2)
        return F.relu(self.fc1(x))
