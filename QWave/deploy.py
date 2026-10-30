"""Deployment helpers: the int8 scaler rule.

Every head in this repository is trained on standardized features: a
StandardScaler fitted on the fit rows (training folds minus the inner
validation split) of the FP32 backbone's embeddings, then the head. When the
backbone is replaced by its real int8 export, every activation sits on an 8-bit
grid. Embedding dimensions whose whole FP32 range is smaller than one grid step
come out of the int8 backbone as a near-constant value that the FP32 scaler,
dividing by their tiny std, turns into a shift of tens to hundreds of standard
deviations. The head then collapses although the raw embedding error is modest
(stage 3c audit of idea 6, 23 Oct 2026). Fake quantization with dynamic scales
does not show this, so a head that looks fine on fake-quant features can still
fail on the deployed backbone.

The rule: a deployed head calibrates its input normalisation on the deployed
backbone. Refit the scaler on the fit rows embedded by the deployed backbone,
keep the head weights as trained. `refit_scaler` does that; `shifted_dims`
reports which dimensions the old scaler would blow up.
"""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def refit_scaler(features_deployed_fit_rows, head):
    """Fit a fresh StandardScaler on the deployed backbone's features and keep the head.

    `features_deployed_fit_rows` are the fit rows of the fold (the same rows the
    training scaler saw), embedded by the backbone that will serve at inference,
    for example the real int8 export. `head` is the trained classifier and is
    returned as it is: its weights were fitted on standardized inputs and stay
    valid once the inputs are standardized against the right backbone.

    Returns `(scaler, head)`. Apply as `head(torch.tensor(scaler.transform(X)))`
    or through `predict`.
    """
    X = np.asarray(features_deployed_fit_rows, dtype=np.float32)
    if X.ndim != 2 or len(X) < 2:
        raise ValueError("features_deployed_fit_rows must be a 2-d array with at least two rows")
    return StandardScaler().fit(X), head


def shifted_dims(scaler_fp32, features_deployed, threshold_sd=1.0):
    """Dimensions the FP32-fit scaler shifts by more than `threshold_sd` on the deployed features.

    Returns an index array, sorted by the size of the per-dimension mean shift in
    the FP32 scaler's z space. Any dimension here means the FP32 scaler is unsafe
    on this backbone and `refit_scaler` is needed.
    """
    X = np.asarray(features_deployed, dtype=np.float32)
    shift = np.abs((X.mean(0) - scaler_fp32.mean_) / scaler_fp32.scale_)
    dims = np.flatnonzero(shift > threshold_sd)
    return dims[np.argsort(-shift[dims])]


@torch.no_grad()
def predict(head, scaler, features):
    """Class predictions of `head` on `features` standardized by `scaler`."""
    head.eval()
    z = torch.tensor(scaler.transform(np.asarray(features, dtype=np.float32)), dtype=torch.float32)
    out = head(z)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.argmax(1).cpu().numpy()
