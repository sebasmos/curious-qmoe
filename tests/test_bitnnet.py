import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from QWave.bitnnet import BitLinear, BitNetExpert

# Rewritten 2026-08-28 against the current API: the old version imported
# MLPBitnet and called quantize_weights()/quantize_activations(), none of
# which exist in QWave/bitnnet.py any more.


def test_bitlinear_forward_shapes():
    in_features, out_features, batch_size = 10, 5, 4
    for num_bits in (4, 8, 16, "bitnet"):
        layer = BitLinear(in_features, out_features, num_bits=num_bits)
        x = torch.randn(batch_size, in_features)
        y = layer(x)
        assert y.shape == (batch_size, out_features), f"Output shape mismatch for num_bits={num_bits}"


def test_bitlinear_quantized_forward_differs_from_fp():
    torch.manual_seed(0)
    in_features, out_features = 32, 8
    x = torch.randn(4, in_features)
    fp = BitLinear(in_features, out_features, num_bits=16)
    q4 = BitLinear(in_features, out_features, num_bits=4)
    q4.weight.data.copy_(fp.weight.data)
    if fp.bias is not None:
        q4.bias.data.copy_(fp.bias.data)
    assert not torch.allclose(fp(x), q4(x)), "4-bit forward should differ from 16-bit on identical weights"


def test_bitnet_expert_forward():
    in_dim, num_classes, batch_size = 16, 4, 3
    for num_bits in (4, "bitnet"):
        model = BitNetExpert(in_dim, num_classes, hidden_sizes=[8, 8], dropout_prob=0.1, num_bits=num_bits)
        model.eval()
        x = torch.randn(batch_size, in_dim)
        y = model(x)
        assert y.shape == (batch_size, num_classes), f"BitNetExpert output shape mismatch for num_bits={num_bits}"
        assert torch.isfinite(y).all(), "BitNetExpert produced non-finite logits"


if __name__ == "__main__":
    test_bitlinear_forward_shapes()
    test_bitlinear_quantized_forward_differs_from_fp()
    test_bitnet_expert_forward()
    print("All bitnnet tests passed")
