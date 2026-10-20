#### QWave BitLinear Module ####
"""One BitNet / k-bit implementation for every model.

This module used to carry its own BitLinear with different BitNet thresholds
(0.05 on weights, sign-only activations) from the one the benchmark and the
MoE experts use. It now re-exports QWave/qmoe_layers.py, so "BitNet" and
"k-bit" mean the same quantizer in single models, MoE experts and tests:
BitNet weights are ternary with zero threshold 0.15 * mean|W - rowmean| and a
per-row scale, activations ternary with zero threshold 0.05 * mean|x|.
"""

from QWave.qmoe_layers import BitLinear, BitNetExpert  # noqa: F401
