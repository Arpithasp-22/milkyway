"""Synthetic star-field generator.

Stars are Gaussian PSFs rendered at *sub-pixel* positions (so sub-pixel
localization can actually be evaluated), streaks are linear trails with a
Gaussian cross-section, and the image has a flat background plus Gaussian noise.

Coordinate convention: pixel centers sit at integer coordinates, x is the column
index and y is the row index.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SimConfig:
    image_size: int = 256

    num_stars: int = 50
    star_peak: tuple = (150.0, 400.0)   # peak counts above background
    psf_sigma: float = 1.2              # pixels
    min_star_separation: float = 10.0   # pixels, keeps stars from merging
    margin: int = 10                    # keep stars away from the border

    num_streaks: int = 5
    streak_peak: tuple = (100.0, 250.0)
    streak_length: tuple = (20, 60)     # pixels
    streak_sigma: float = 1.0           # cross-section width, pixels

    background: float = 20.0
    noise_sigma: float = 3.0


@dataclass
class Scene:
    image: np.ndarray    # (H, W) float32
    stars: np.ndarray    # (N, 3): x, y, peak
    streaks: np.ndarray  # (M, 5): x1, y1, x2, y2, peak


def _place_stars(cfg, rng):
    low, high = cfg.margin, cfg.image_size - 1 - cfg.margin
    points = []
    for _ in range(20000):
        if len(points) == cfg.num_stars:
            break
        p = rng.uniform(low, high, size=2)
        if all(np.hypot(*(p - q)) >= cfg.min_star_separation for q in points):
            points.append(p)
    if len(points) < cfg.num_stars:
        raise ValueError(
            "Could not place all stars; lower num_stars or min_star_separation."
        )
    return np.array(points)


def _segment_distance(xx, yy, x1, y1, x2, y2):
    """Distance from every pixel to the line segment (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    t = ((xx - x1) * dx + (yy - y1) * dy) / (dx * dx + dy * dy)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(xx - (x1 + t * dx), yy - (y1 + t * dy))


def _place_streak(cfg, rng):
    """Random in-frame segment with a length drawn from cfg.streak_length."""
    last = cfg.image_size - 1
    for _ in range(1000):
        x1, y1 = rng.uniform(0, last, size=2)
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(*cfg.streak_length)
        x2 = x1 + length * np.cos(angle)
        y2 = y1 + length * np.sin(angle)
        if 0 <= x2 <= last and 0 <= y2 <= last:
            return x1, y1, x2, y2
    raise ValueError("Could not place a streak inside the image.")


def generate_scene(cfg=None, seed=0):
    """Return a Scene with ground truth. The same seed gives the same scene."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:cfg.image_size, 0:cfg.image_size].astype(np.float64)

    image = np.full((cfg.image_size, cfg.image_size), cfg.background)

    star_xy = _place_stars(cfg, rng)
    star_peaks = rng.uniform(*cfg.star_peak, size=cfg.num_stars)
    for (x, y), peak in zip(star_xy, star_peaks):
        r2 = (xx - x) ** 2 + (yy - y) ** 2
        image += peak * np.exp(-r2 / (2 * cfg.psf_sigma ** 2))

    streaks = []
    for _ in range(cfg.num_streaks):
        x1, y1, x2, y2 = _place_streak(cfg, rng)
        peak = rng.uniform(*cfg.streak_peak)
        d = _segment_distance(xx, yy, x1, y1, x2, y2)
        image += peak * np.exp(-d ** 2 / (2 * cfg.streak_sigma ** 2))
        streaks.append((x1, y1, x2, y2, peak))

    image += rng.normal(0.0, cfg.noise_sigma, image.shape)
    image = np.clip(image, 0, None).astype(np.float32)

    return Scene(
        image=image,
        stars=np.column_stack([star_xy, star_peaks]),
        streaks=np.array(streaks).reshape(-1, 5),
    )
