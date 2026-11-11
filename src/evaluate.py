"""Score detections against the simulator's ground truth."""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .simulate import _segment_distance

METHODS = ("binary", "weighted", "subpixel")


@dataclass
class TrialResult:
    # Star detection (round blobs matched one-to-one to ground-truth stars)
    n_true_stars: int
    n_star_detections: int
    n_matched_stars: int
    # Localization error (pixels) for matched stars, per centroid method
    errors: dict = field(default_factory=dict)
    # Streak detection
    n_true_streaks: int = 0
    n_streak_detections: int = 0
    n_streaks_found: int = 0
    n_streak_detections_correct: int = 0
    # Fitted Gaussian widths for the sigma analysis
    star_sigmas: list = field(default_factory=list)    # (sigma_x, sigma_y)
    streak_sigmas: list = field(default_factory=list)
    # (detection, ground-truth index) pairs, kept for plotting
    matches: list = field(default_factory=list)


def match_points(pred, truth, max_dist=3.0):
    """One-to-one matching (Hungarian) of predicted to true points within max_dist."""
    if len(pred) == 0 or len(truth) == 0:
        return []
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    dist = np.hypot(pred[:, None, 0] - truth[None, :, 0],
                    pred[:, None, 1] - truth[None, :, 1])
    cost = np.where(dist <= max_dist, dist, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if dist[r, c] <= max_dist]


def evaluate_trial(scene, detections, max_dist=3.0):
    stars = [d for d in detections if not d.is_streak]
    streak_dets = [d for d in detections if d.is_streak]

    # ---- stars ---------------------------------------------------------------
    truth_xy = scene.stars[:, :2]
    pairs = match_points([d.centroid_weighted for d in stars], truth_xy, max_dist)

    errors = {m: [] for m in METHODS}
    matches = []
    for di, ti in pairs:
        det, (tx, ty) = stars[di], truth_xy[ti]
        errors["binary"].append(np.hypot(det.centroid_binary[0] - tx, det.centroid_binary[1] - ty))
        errors["weighted"].append(np.hypot(det.centroid_weighted[0] - tx, det.centroid_weighted[1] - ty))
        if det.fit is not None:
            errors["subpixel"].append(np.hypot(det.fit.x - tx, det.fit.y - ty))
        matches.append((det, ti))

    result = TrialResult(
        n_true_stars=len(truth_xy),
        n_star_detections=len(stars),
        n_matched_stars=len(pairs),
        errors={m: np.array(v) for m, v in errors.items()},
        matches=matches,
    )
    result.star_sigmas = [(stars[di].fit.sigma_x, stars[di].fit.sigma_y)
                          for di, _ in pairs if stars[di].fit is not None]

    # ---- streaks -------------------------------------------------------------
    result.n_true_streaks = len(scene.streaks)
    result.n_streak_detections = len(streak_dets)
    found = set()
    for det in streak_dets:
        cx, cy = det.centroid_weighted
        dists = [float(_segment_distance(cx, cy, *s[:4])) for s in scene.streaks]
        if dists and min(dists) <= max_dist:
            result.n_streak_detections_correct += 1
            found.add(int(np.argmin(dists)))
        if det.fit is not None:
            result.streak_sigmas.append((det.fit.sigma_x, det.fit.sigma_y))
    result.n_streaks_found = len(found)
    return result


def _rmse(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else float("nan")


def summarize(results):
    """Pool trial results into one dictionary of headline metrics."""
    tp = sum(r.n_matched_stars for r in results)
    det = sum(r.n_star_detections for r in results)
    truth = sum(r.n_true_stars for r in results)

    pooled = {m: np.concatenate([r.errors[m] for r in results]) for m in METHODS}
    star_sig = np.array([s for r in results for s in r.star_sigmas]).reshape(-1, 2)
    streak_sig = np.array([s for r in results for s in r.streak_sigmas]).reshape(-1, 2)

    sd = sum(r.n_streak_detections for r in results)
    return {
        "trials": len(results),
        "star_recall": tp / truth if truth else float("nan"),
        "star_precision": tp / det if det else float("nan"),
        "rmse_px": {m: _rmse(pooled[m]) for m in METHODS},
        "streak_recall": sum(r.n_streaks_found for r in results)
        / max(sum(r.n_true_streaks for r in results), 1),
        "streak_precision": sum(r.n_streak_detections_correct for r in results) / sd if sd else float("nan"),
        "star_sigma_mean": star_sig.mean(axis=0).tolist() if len(star_sig) else None,
        "streak_sigma_mean": streak_sig.mean(axis=0).tolist() if len(streak_sig) else None,
    }
