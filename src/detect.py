"""Classical, deterministic detection and localization.

1. Robust background/noise estimate (median and MAD).
2. Threshold, then connected-component labeling.
3. Three centroid estimates per blob: binary (unweighted), intensity-weighted,
   and a sub-pixel 2D Gaussian fit.
4. Blob elongation (from second moments) separates round stars from streaks.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit

EIGHT_CONNECTED = np.ones((3, 3), dtype=int)


@dataclass
class DetectConfig:
    threshold_sigma: float = 5.0     # threshold = background + k * noise sigma
    min_area: int = 4                # pixels
    patch_radius: int = 5            # half-size of the Gaussian-fit window
    streak_elongation: float = 2.0   # sqrt(major/minor variance) above this = streak


@dataclass
class GaussianFit:
    x: float
    y: float
    sigma_x: float
    sigma_y: float
    amplitude: float


@dataclass
class Detection:
    area: int
    centroid_binary: tuple      # unweighted mean of the blob's pixel coordinates
    centroid_weighted: tuple    # background-subtracted intensity-weighted mean
    elongation: float           # 1.0 = round, larger = more elongated
    is_streak: bool
    fit: Optional[GaussianFit]  # None if the Gaussian fit failed

    @property
    def centroid_subpixel(self):
        return (self.fit.x, self.fit.y) if self.fit else None


def estimate_background(image):
    """Median background and robust noise sigma (1.4826 * MAD)."""
    bg = float(np.median(image))
    sigma = float(1.4826 * np.median(np.abs(image - bg)))
    return bg, max(sigma, 1e-6)


def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = coords
    return (
        amplitude * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2)
                             + (y - y0) ** 2 / (2 * sigma_y ** 2)))
        + offset
    ).ravel()


def fit_gaussian(image, cx, cy, patch_radius=5):
    """Fit an axis-aligned 2D Gaussian to the patch around (cx, cy)."""
    h, w = image.shape
    x0, y0 = int(round(cx)), int(round(cy))
    x_min, x_max = max(0, x0 - patch_radius), min(w, x0 + patch_radius + 1)
    y_min, y_max = max(0, y0 - patch_radius), min(h, y0 + patch_radius + 1)

    patch = image[y_min:y_max, x_min:x_max].astype(np.float64)
    if patch.size < 7:
        return None
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]

    offset0 = float(np.median(patch))
    p0 = (max(patch.max() - offset0, 1.0), cx, cy, 1.5, 1.5, offset0)
    max_sigma = float(2 * patch_radius + 1)
    lower = (0.0, x_min, y_min, 0.3, 0.3, -np.inf)
    upper = (np.inf, x_max - 1, y_max - 1, max_sigma, max_sigma, np.inf)

    try:
        popt, _ = curve_fit(
            gaussian_2d, (xx, yy), patch.ravel(),
            p0=p0, bounds=(lower, upper), maxfev=2000,
        )
    except (RuntimeError, ValueError):
        return None

    amplitude, x, y, sx, sy, _ = popt
    return GaussianFit(float(x), float(y), float(sx), float(sy), float(amplitude))


def _blob_shape(xs, ys, weights):
    """Weighted centroid and elongation from the blob's second moments."""
    total = weights.sum()
    if total <= 0:
        weights = np.ones_like(weights)
        total = weights.sum()
    mx, my = (weights * xs).sum() / total, (weights * ys).sum() / total
    dx, dy = xs - mx, ys - my
    cov = np.array([
        [(weights * dx * dx).sum(), (weights * dx * dy).sum()],
        [(weights * dx * dy).sum(), (weights * dy * dy).sum()],
    ]) / total
    minor, major = np.sort(np.linalg.eigvalsh(cov))
    elongation = float(np.sqrt(major / max(minor, 1e-6)))
    return (float(mx), float(my)), elongation


def detect(image, cfg=None):
    """Detect blobs and return a list of Detection objects."""
    cfg = cfg or DetectConfig()
    bg, sigma = estimate_background(image)
    labeled, _ = ndimage.label(image > bg + cfg.threshold_sigma * sigma,
                               structure=EIGHT_CONNECTED)

    detections = []
    for label, sl in enumerate(ndimage.find_objects(labeled), start=1):
        mask = labeled[sl] == label
        area = int(mask.sum())
        if area < cfg.min_area:
            continue

        ys, xs = np.nonzero(mask)
        ys, xs = ys + sl[0].start, xs + sl[1].start
        weights = np.clip(image[ys, xs].astype(np.float64) - bg, 0, None)

        centroid_weighted, elongation = _blob_shape(xs.astype(float), ys.astype(float), weights)
        centroid_binary = (float(xs.mean()), float(ys.mean()))

        detections.append(Detection(
            area=area,
            centroid_binary=centroid_binary,
            centroid_weighted=centroid_weighted,
            elongation=elongation,
            is_streak=elongation >= cfg.streak_elongation,
            fit=fit_gaussian(image, *centroid_weighted, patch_radius=cfg.patch_radius),
        ))
    return detections
