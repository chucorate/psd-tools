import logging
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from PIL import Image

from psd_tools.api.layers import Layer
from psd_tools.constants import ColorMode
from psd_tools.composite._compat import require_scipy
from psd_tools.composite.blend import _lum, _cmyk2rgb
from psd_tools.composite.utils import rgb2hsl, hsl2rgb, hsl2hsv, hsv2hsl
from psd_tools.api.adjustments import (
    BrightnessContrast,
    Levels, 
    Curves, 
    Exposure, 
    HueSaturation,
    Invert,
    Posterize,
    Threshold,
)

logger = logging.getLogger(__name__)


@require_scipy
def apply_brightnesscontrast(
    layer: BrightnessContrast, 
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB]
) -> np.ndarray:

    use_legacy: bool  = layer.use_legacy
    b: float = layer.brightness / 150.0 
    c: float   = layer.contrast / 100.0
        
    size = _get_mode_info(layer)
    t = np.linspace(0, 1, size, dtype=np.float32)
        
    if use_legacy: # these layers are skipped during composing as they are recognized as PixelLayers with no bounding box
        return img
    
    # TODO: improve brightness function accuracy
    # the non-legacy adjustment was determined using reverse engineering and tuning parameters, might be slightly off
    # check brightness curve: https://www.desmos.com/calculator/4fg6glxzqj
    
    # contrast 
    x = np.array([0.0, 63.0, 191.0, 255.0]) / 255.0
    y = np.array([0.0, 63.0 - c * 25.0, 191.0 + c * 25.0, 255.0]) / 255.0
    contrast_spline = CubicSpline(x, y, bc_type="natural")(t)

    # brightness 
    a1, a2, a3, a4, a5 = 1.65, -1.0, 1.96, 1.0, 1.00
    r1, r2, r3, r4, r5 = 0.35, 10.0, 0.4, 4.0, 1.25

    def pol(a,x,r): return a * np.power(x, r)

    h = 0.5 * (abs(b) * (pol(a1,t,r1) +  pol(a2,t,r2)) + (1-abs(b)) * (pol(a3,t,r3) +  pol(a4,t,r4)) + pol(a5,t,r5))
    brightness_spline = b * t * (1-t) * h

    # a parametric transformation rotates the brightness spline function 45° degrees
    x_rotated = t - brightness_spline
    y_rotated = contrast_spline + brightness_spline 

    # interpolate to output {size} points
    lut = np.interp(t, x_rotated, y_rotated)
    lut = lut.clip(0.0, 1.0).astype(np.float32)

    channel_id = 1 if colormode == ColorMode.GRAYSCALE else 0

    return apply_luts({channel_id: lut}, img, colormode)


def apply_levels(
    layer: Levels, 
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB]
) -> np.ndarray:

    levels_data = layer.data
    size = _get_mode_info(layer)

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

    return apply_luts(luts, img, colormode)


@require_scipy
def apply_curves(
    layer: Curves, 
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB]
) -> np.ndarray:

    curves_data = layer.extra
    info_dict = {data.channel_id: data.points for data in curves_data}

    size = _get_mode_info(layer)

    luts: dict[int, NDArray[np.float32]] = {}

    for channel_id, points in info_dict.items():
        if len(points) < 2: continue
        
        x = np.array([p[1] for p in points]) / 255.0
        y = np.array([p[0] for p in points]) / 255.0

        cs = CubicSpline(x, y, bc_type="natural")
        
        x_min, x_max = x[0], x[-1]
        t = np.linspace(0.0, 1.0, size, dtype=np.float32)
        t_clamped = np.clip(t, x_min, x_max)

        lut = cs(t_clamped).clip(0.0, 1.0).astype(np.float32)
        luts[channel_id] = lut

    return apply_luts(luts, img, colormode)


def apply_exposure(
    layer: Exposure, 
    img: np.ndarray, 
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB]
) -> np.ndarray:

    exposure: float = layer.exposure
    offset: float   = layer.exposure_offset
    gamma: float    = layer.gamma

    size = _get_mode_info(layer)
    color_gamma = 1.8 if colormode == ColorMode.GRAYSCALE else 2.2
    values = np.linspace(0.0, 1.0, size, dtype=np.float32)

    factor = np.pow(2.0, exposure/color_gamma)
    lut = (values * factor).clip(0.0, 1.0)

    lut = (np.power(lut, color_gamma) + offset).clip(0.0, 1.0)
    lut = np.power(lut, 1.0/(color_gamma * gamma))

    lut = lut.clip(0.0, 1.0).astype(np.float32)

    channel_id = 1 if colormode == ColorMode.GRAYSCALE else 0

    return apply_luts({channel_id: lut}, img, colormode)


def apply_huesaturation(
    layer: HueSaturation, 
    img: np.ndarray, 
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB]
) -> np.ndarray:

    if colormode != ColorMode.RGB:
        # TODO: add CMYK support
        return img

    if layer.enable_colorization:
        colorization = layer.colorization
        return _huesaturation_colorize(img, colorization) if colorization != (0,0,0) else img

    color_ranges = [(hue_range, hsl_tuple) for hue_range, hsl_tuple in layer.data if hsl_tuple != (0,0,0)]
    master = layer.master
    return _huesaturation(img, color_ranges, master) if color_ranges or master != (0,0,0) else img


def _norm_hsl(color: tuple[int, int, int]) -> tuple[float, float, float]:
    h, s, l = color
    return (h / 360.0), s / 100.0, l / 100.0


def _huesaturation_colorize(
    img: np.ndarray, 
    colorize_tuple: tuple[int, int, int]
) -> np.ndarray:
    h, s, l = _norm_hsl(colorize_tuple)
    h = h % 1.0

    hsl = rgb2hsl(img)
    L = hsl[..., 2:3]

    out = np.empty_like(hsl)
    out[..., 0:1] = h
    out[..., 1:2] = s
    out[..., 2:3] = L*(1.0 - l) + l if l > 0 else L*(1.0 + l) if l < 0 else L
    return hsl2rgb(out)


def _huesaturation(
    img: np.ndarray, 
    color_ranges: list[tuple[tuple[int], tuple[int, int, int]]], 
    master_tuple: tuple[int, int, int]
) -> np.ndarray:

    # TODO: improve saturation transition between color ranges
    saturation_easing = lambda x, p: 1 - (1-x)**(1.5*(1/(1-p**4+1e-2))) if p > 0 else x # might be slightly off

    dark_easing = lambda x, p: 1 + (1/(p+1e-6) - 1)/(x - 1/(p+1e-6) + 1e-6) # fits exactly

    # master lightness is applied before converting to hsl
    hM, sM, lM = _norm_hsl(master_tuple)
    img = img*(1.0 - lM) + lM if lM > 0 else img*(1.0 + lM) if lM < 0 else img

    hsl = rgb2hsl(img)
    H_base = hsl[..., 0:1]
    sat_mask = hsl[..., 1:2] > 1e-6

    h_mask = np.zeros_like(H_base)
    s_mask = np.zeros_like(H_base)
    l_mask = np.zeros_like(H_base)
    s_contrib = []

    for hue_range_tuple, color_tuple in color_ranges:
        h, s, l = _norm_hsl(color_tuple)
        hue_range_mask = _get_huesaturation_range_mask(H_base, hue_range_tuple) * sat_mask

        # h_mask and l_mask depend on loop order, s_mask doesn't
        h_mask += h*hue_range_mask.clip(-1.0, 1.0)
        l_mask += l*hue_range_mask.clip(-1.0, 1.0)

        contrib = s * saturation_easing(hue_range_mask, s)
        s_contrib.append(contrib)
        
    if s_contrib:
        all_s = np.stack(s_contrib, axis=0)
        idx_max = np.argmax(all_s, axis=0)
        s_mask = np.take_along_axis(all_s, np.expand_dims(idx_max, axis=0), axis=0).squeeze(0)

    # lightness is applied in HSV space
    hsv = hsl2hsv(hsl)
    SV, LV = hsv[..., 1:2], hsv[..., 2:3]
    hsv[..., 2:3] = np.where(l_mask < 0, LV*((1+l_mask) - l_mask*(1 - SV)), LV)
    hsv[..., 1:2] = np.where(l_mask > 0, SV*(1-l_mask).clip(0.0, 1.0), dark_easing(np.abs(l_mask), SV).clip(0.0, 1.0))
    hsl = hsv2hsl(hsv)
    
    # hue and saturation are applied in HSL space
    hsl[..., 0:1] = (hsl[..., 0:1] + h_mask + hM) % 1.0

    S = hsl[..., 1:2]
    hsl[..., 1:2] = (S /(1-sM + 1e-6)).clip(0.0, 1.0) if sM > 0 else (S *(1+sM)).clip(0.0, 1.0) if sM < 0 else S
    hsl[..., 1:2] = np.where(s_mask > 0, 
        (hsl[..., 1:2] / (1 - s_mask + 1e-6)).clip(0.0, 1.0), 
        (hsl[..., 1:2] * (1 + s_mask)).clip(0.0, 1.0),
    )

    return hsl2rgb(hsl)


def _get_huesaturation_range_mask(hue: np.ndarray, color_range: tuple[int]) -> np.ndarray:
    a, b, c, d = np.array(color_range) / 360.0

    mask_curve = lambda x: x

    hue0 = (hue - a) % 1.0
    b0   = (b - a) % 1.0
    c0   = (c - a) % 1.0
    d0   = (d - a) % 1.0

    mask = np.zeros_like(hue0)

    center = (hue0 >= b0) & (hue0 <= c0)
    mask[center] = 1.0

    left = (hue0 >= 0.0) & (hue0 < b0)
    mask[left] = mask_curve(hue0[left]) / mask_curve(b0)

    right = (hue0 > c0) & (hue0 <= d0)
    mask[right] = mask_curve(d0 - hue0[right]) / mask_curve(d0 - c0)

    return mask


def apply_invert(
        layer: Invert, 
        img: np.ndarray, 
        colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB]
    ) -> np.ndarray:

    return 1.0 - img


def apply_posterize(
    layer: Posterize, 
    img: np.ndarray, 
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB]
) -> np.ndarray:

    size = _get_mode_info(layer)
    levels = layer.posterize

    values = np.linspace(0.0, 1.0, size, dtype=np.float32)

    lut = (np.floor(levels * values) / (levels-1)).astype(np.float32)
    channel_id = 1 if colormode == ColorMode.GRAYSCALE else 0

    return apply_luts({channel_id: lut}, img, colormode)


def apply_threshold(
    layer: Threshold, 
    img: np.ndarray, 
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB]
) -> np.ndarray:

    size = _get_mode_info(layer) - 1 
    trunc_function = np.round if size < 256 else np.floor

    threshold = (layer.threshold - 1) / 255.0 * size
    luminance = trunc_function(_get_luminance(img, colormode) * size) 
    
    filtered = (luminance > threshold).astype(np.float32) 

    if colormode == ColorMode.CMYK:
        h, w, _ = filtered.shape
        out = np.ones((h, w, 4), dtype=np.float32)
        out[..., 3:4] = 1.0 - filtered
        return out
    
    elif colormode == ColorMode.RGB:
        out = np.repeat(filtered, 3, axis=2)
        return out
    
    return filtered


def apply_luts(luts: dict[int, NDArray[np.float32]], img: np.ndarray, colormode: ColorMode) -> np.ndarray:
    assert img.ndim == 3
    out = img.copy()
    n_channels = ColorMode.channels(colormode)
    channels = range(1, n_channels+1)
    
    # individual adjustments get applied independently of each other, then the master lut its applied if exists
    for channel_id in channels:
        if channel_id in luts:
            lut = luts[channel_id]
            out[:, :, channel_id-1] = _apply_lut(img[:, :, channel_id-1], lut)

    if not colormode == ColorMode.GRAYSCALE and 0 in luts:
        out = _apply_lut(out, luts[0])
  
    return out


def _apply_lut(values: np.ndarray, lut: np.ndarray) -> np.ndarray:
    size = lut.shape[0]

    if size <= 2**16:
        depth = size
        values = (np.floor(values * depth) / depth).clip(0.0, 1.0)

    xp = np.linspace(0.0, 1.0, size, dtype=np.float32)
    return np.interp(values, xp, lut).astype(np.float32)
    

def _get_mode_info(layer: Layer) -> int:
    bits = layer._psd.depth
    size = min(2**bits, 65536) 
    logger.debug(f"Size = {size}")
    return size 


def _get_luminance(
    img: np.ndarray, 
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB]
) -> np.ndarray:
    if colormode == ColorMode.RGB:
        return _lum(img)
    elif colormode == ColorMode.GRAYSCALE:
        return img[..., 0:1]
    elif colormode == ColorMode.CMYK:
        lab = Image.fromarray((img * 255).astype(np.uint8), "CMYK").convert("LAB")
        return (np.asarray(lab)[..., 0:1] / 255.0).clip(0.0, 1.0) # somewhat inaccurate


# wip
"""Adjustment function table."""
ADJUSTMENT_FUNC = {
    "brightnesscontrast": apply_brightnesscontrast,
    "levels": apply_levels,
    "curves": apply_curves,
    "exposure": apply_exposure,
    "huesaturation": apply_huesaturation,
    "invert": apply_invert,
    "posterize": apply_posterize,
    "threshold": apply_threshold,
}