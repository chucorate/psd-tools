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


def apply_levels(layer: Levels, img: np.ndarray) -> np.ndarray:
    mode = _get_mode(layer)
    if mode is None:
        return img
    
    levels_data = layer.data
    size = _get_size(layer)

    luts: dict[int, NDArray[np.float32]] = {}

    for channel_id, channel_data in enumerate(levels_data):        
        in_black: float  = channel_data.input_floor / 255.0
        in_white: float  = channel_data.input_ceiling / 255.0
        gamma: float     = channel_data.gamma / 100.0
        out_black: float = channel_data.output_floor / 255.0
        out_white: float = channel_data.output_ceiling / 255.0

        t = np.linspace(0, 1, size, dtype=np.float32)

        # input adjustments
        scale = (in_white - in_black) if in_white != in_black else 1.0
        out = (t - in_black) / scale
        out = out.clip(0.0, 1.0)
            
        # gamma midtone adjustment
        out = np.power(out, 1.0 / gamma)
        out = out.clip(0.0, 1.0)
            
        # output adjustments
        out = out * (out_white - out_black) + out_black
        lut = out.clip(0.0, 1.0).astype(np.float32)

        luts[channel_id] = lut

    return apply_luts(luts, img, mode)


def apply_curves(layer: Curves, img: np.ndarray) -> np.ndarray:
    mode = _get_mode(layer)
    if mode is None:
        return img
    
    curves_data = layer.extra
    info_dict = {data.channel_id: data.points for data in curves_data}

    size = _get_size(layer)

    luts: dict[int, NDArray[np.float32]] = {}

    for channel_id, points in info_dict.items():
        if len(points) < 2: continue
        
        x = np.array([p[1] for p in points]) / 255.0
        y = np.array([p[0] for p in points]) / 255.0

        cs = CubicSpline(x, y, bc_type="natural")
        t = np.linspace(0.0, 1.0, size, dtype=np.float32)
        lut = cs(t).clip(0.0, 1.0).astype(np.float32)

        luts[channel_id] = lut

    return apply_luts(luts, img, mode)


def apply_exposure(layer: Exposure, img: np.ndarray) -> np.ndarray:
    mode = _get_mode(layer)
    if mode is None:
        return img
    
    exposure: float = layer.exposure
    offset: float   = layer.exposure_offset
    gamma: float    = layer.gamma

    size = _get_size(layer)

    values = np.linspace(0.0, 1.0, size, dtype=np.float32)

    factor = math.pow(2.0, exposure/2.2)
    lut = (values * factor).clip(0.0, 1.0)

    lut = (np.power(lut, 2.2) + offset).clip(0.0, 1.0)
    lut = np.power(lut, 1.0/(2.2 * gamma))

    lut = lut.clip(0.0, 1.0).astype(np.float32)

    return apply_luts({0: lut}, img, mode)


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


def _get_mode(layer: Layer) -> Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB] | None:
    mode = layer._psd.color_mode
    if mode in (ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB):
        return mode


# wip
"""Adjustment function table."""
ADJUSTMENT_FUNC = {
    "levels": apply_levels,
    "curves": apply_curves,
}