"""Utility functions for composite operations."""

from typing import Union, overload

import numpy as np
from numpy.typing import NDArray

from psd_tools.api.layers import Layer
from psd_tools.constants import Tag


def divide(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.floating]:
    """Safe division for color ops."""
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 1.0
    return c


def intersect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Calculate intersection of two bounding boxes."""
    inter = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    if inter[0] >= inter[2] or inter[1] >= inter[3]:
        return (0, 0, 0, 0)
    return inter


def has_fill(layer: Layer) -> bool:
    """Check if layer has fill settings."""
    FILL_TAGS = (
        Tag.SOLID_COLOR_SHEET_SETTING,
        Tag.PATTERN_FILL_SETTING,
        Tag.GRADIENT_FILL_SETTING,
        Tag.VECTOR_STROKE_CONTENT_DATA,
    )
    return any(tag in layer.tagged_blocks for tag in FILL_TAGS)


@overload
def union(backdrop: float, source: float) -> float: ...


@overload
def union(
    backdrop: NDArray[np.floating], source: NDArray[np.floating]
) -> NDArray[np.floating]: ...


@overload
def union(backdrop: float, source: NDArray[np.floating]) -> NDArray[np.floating]: ...


@overload
def union(backdrop: NDArray[np.floating], source: float) -> NDArray[np.floating]: ...


def union(
    backdrop: Union[float, NDArray[np.floating]],
    source: Union[float, NDArray[np.floating]],
) -> Union[float, NDArray[np.floating]]:
    """Generalized union of shape."""
    return backdrop + source - (backdrop * source)


def clip(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Clip between [0, 1]."""
    return np.clip(x, 0.0, 1.0)


def rgb2hsl(img: np.ndarray) -> np.ndarray:
    ch_min = np.min(img, axis = 2)
    ch_max = np.max(img, axis = 2)

    delta, total = ch_max-ch_min, ch_max+ch_min
    non_zero = delta > 1e-6

    L = total / 2

    S = np.zeros_like(L)
    S[non_zero] = delta[non_zero] / (1 - np.abs(2 * L[non_zero] - 1))

    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    idx_R = (ch_max == R) & non_zero
    idx_G = (ch_max == G) & non_zero
    idx_B = (ch_max == B) & non_zero

    H = np.zeros_like(L)
    H[idx_R] = ((G[idx_R] - B[idx_R]) / delta[idx_R]) % 6.0
    H[idx_G] = ((B[idx_G] - R[idx_G]) / delta[idx_G]) + 2.0
    H[idx_B] = ((R[idx_B] - G[idx_B]) / delta[idx_B]) + 4.0
    H /= 6.0

    return np.stack([H, S, L], axis = 2)


def hsl2rgb(img: np.ndarray) -> np.ndarray:
    H, S, L = img[..., 0], img[..., 1], img[..., 2]
    
    if np.all(S < 1e-6):
        return np.repeat(L[..., None], 3, axis = 2)
    
    C = (1 - np.abs(2 * L - 1)) * S
    H_ = H * 6.0
    X = C * (1 - np.abs(H_ % 2 - 1))

    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    mask0 = (H_ < 1)
    mask1 = (1 <= H_) & (H_ < 2)
    mask2 = (2 <= H_) & (H_ < 3)
    mask3 = (3 <= H_) & (H_ < 4)
    mask4 = (4 <= H_) & (H_ < 5)
    mask5 = (5 <= H_) 

    R[mask0], G[mask0], B[mask0] = C[mask0], X[mask0], 0
    R[mask1], G[mask1], B[mask1] = X[mask1], C[mask1], 0
    R[mask2], G[mask2], B[mask2] = 0, C[mask2], X[mask2]
    R[mask3], G[mask3], B[mask3] = 0, X[mask3], C[mask3]
    R[mask4], G[mask4], B[mask4] = X[mask4], 0, C[mask4]
    R[mask5], G[mask5], B[mask5] = C[mask5], 0, X[mask5]

    m = L - C / 2.0
    rgb = np.stack([R + m, G + m, B + m], axis=2).clip(0.0, 1.0)
    
    return rgb


def hsl2hsv(img: np.ndarray) -> np.ndarray:
    H, SL, L = img[..., 0], img[..., 1], img[..., 2]

    V = L + SL * np.minimum(L, 1.0 - L)

    SV = np.zeros_like(V)
    mask = V > 1e-12
    SV[mask] = 2.0 * (1.0 - L[mask] / V[mask])

    return np.stack([H, SV, V], axis=-1).clip(0.0, 1.0)


def hsv2hsl(img: np.ndarray) -> np.ndarray:
    H, SV, V = img[..., 0], img[..., 1], img[..., 2]

    L = V * (1.0 - SV / 2.0)

    SL = np.zeros_like(L)
    denom = np.minimum(L, 1.0 - L)
    mask = denom > 1e-12
    SL[mask] = (V[mask] - L[mask]) / denom[mask]

    return np.stack([H, SL, L], axis=-1).clip(0.0, 1.0)

