import logging
from typing import Any, Literal

import math
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from psd_tools.api.layers import Layer
from psd_tools.constants import Tag, ColorMode
from psd_tools.api.adjustments import (
    BrightnessContrast,
    Levels, 
    Curves, 
    Exposure, 
)

logger = logging.getLogger(__name__)


def apply_curves(layer: Curves, img: np.ndarray) -> np.ndarray:
    mode = layer._psd.color_mode
    if mode not in (ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB):
        return img
    
    info = layer.tagged_blocks.get_data(Tag.CURVES).extra
    info_dict = {data.channel_id: data.points for data in info}

    size = _get_size(layer)

    luts: dict[int, NDArray[np.float32]] = {}

    for channel_id, points in info_dict.items():
        if len(points) < 2: continue
        
        x = np.array([p[1] for p in points]) / 255.0
        y = np.array([p[0] for p in points]) / 255.0

        cs = CubicSpline(x, y, bc_type="natural")
        t = np.linspace(0, 1, size, dtype=np.float32)
        lut = cs(t).clip(0, 1).astype(np.float32)

        luts[channel_id] = lut

    return apply_luts(luts, img, mode)


def apply_luts(luts: dict[int, NDArray[np.float32]], img: np.ndarray, color_mode: ColorMode) -> np.ndarray:
    assert img.ndim == 3
    out = img.copy()
    n_channels = ColorMode.channels(color_mode)
    channels = range(1, n_channels+1)
    
    # individual adjustments get applied independently of each other, then the master lut its applied if exists
    for channel_id in channels:
        if channel_id in luts:
            lut = luts[channel_id]
            out[:, :, channel_id-1] = _apply_lut(img[:, :, channel_id-1], lut)

    if not color_mode == ColorMode.GRAYSCALE and 0 in luts:
        out = _apply_lut(out, luts[0])
  
    return out


def _apply_lut(values: np.ndarray, lut: np.ndarray) -> np.ndarray:
    xp = np.linspace(0.0, 1.0, lut.shape[0], dtype=np.float32)
    return np.interp(values, xp, lut).astype(np.float32)


def _get_size(layer: Layer) -> int:
    return min(2**layer._psd.depth, 4096)


# wip
"""Adjustment function table."""
ADJUSTMENT_FUNC = {
    "curves": apply_curves,
}