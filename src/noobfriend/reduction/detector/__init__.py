"""Per-frame, mode-agnostic detector cleanups on ``(data, err, dq)`` arrays.

1/f striping removal, outlier / bad-pixel flagging and 2-D self-background
subtraction -- the instrumental corrections shared by imaging and grism frames.
The source-mask helper they build on lives in the internal :mod:`._masking`.
"""

from noobfriend.reduction.detector._background import subtract_background
from noobfriend.reduction.detector._badpixel import flag_outlier_pixels
from noobfriend.reduction.detector._oneoverf import subtract_oneoverf

__all__ = [
    "flag_outlier_pixels",
    "subtract_background",
    "subtract_oneoverf",
]
