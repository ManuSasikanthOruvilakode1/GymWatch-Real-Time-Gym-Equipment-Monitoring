"""Draw the live status overlay onto frames."""

from __future__ import annotations

import cv2
import numpy as np

# BGR
_COLOURS = {"in_use": (60, 60, 230), "free": (90, 190, 60), "unknown": (150, 150, 150)}
_TEXT = {"in_use": "IN USE", "free": "FREE", "unknown": "WARMING UP"}


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def annotate(
    rgb: np.ndarray,
    *,
    name: str,
    state: str,
    p_raw: float,
    p_smooth: float,
    state_age_s: float,
    roi: tuple[float, float, float, float] | None,
    width: int,
) -> np.ndarray:
    """Return a resized BGR frame with ROI box, status banner and probability bar."""
    h, w = rgb.shape[:2]
    scale = width / w
    bgr = cv2.cvtColor(
        cv2.resize(rgb, (width, round(h * scale)), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR
    )
    fh, fw = bgr.shape[:2]
    colour = _COLOURS.get(state, _COLOURS["unknown"])

    if roi is not None:
        x0, y0, x1, y1 = roi
        cv2.rectangle(
            bgr, (int(x0 * fw), int(y0 * fh)), (int(x1 * fw) - 1, int(y1 * fh) - 1), colour, 2
        )

    banner_h = 54
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (fw, banner_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.7, bgr, 0.3, 0, dst=bgr)
    cv2.putText(bgr, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    label = _TEXT.get(state, state.upper())
    if state != "unknown":
        label += f"  {_fmt_duration(state_age_s)}"
    cv2.putText(bgr, label, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

    # Probability bar along the bottom: filled = smoothed, tick = raw frame estimate.
    bar_y, bar_h = fh - 12, 8
    cv2.rectangle(bgr, (0, bar_y), (fw, bar_y + bar_h), (40, 40, 40), -1)
    cv2.rectangle(bgr, (0, bar_y), (int(fw * p_smooth), bar_y + bar_h), colour, -1)
    x_raw = int(fw * p_raw)
    cv2.line(bgr, (x_raw, bar_y - 4), (x_raw, bar_y + bar_h + 2), (255, 255, 255), 2)
    return bgr
