"""Tests that pin down the claims the README makes about the pipeline."""
import numpy as np

from src.detect import DetectConfig, detect, fit_gaussian
from src.evaluate import evaluate_trial, match_points, summarize
from src.simulate import SimConfig, generate_scene


def run_trials(n=5, **sim_kwargs):
    cfg = SimConfig(**sim_kwargs)
    results = []
    for seed in range(n):
        scene = generate_scene(cfg, seed=seed)
        results.append(evaluate_trial(scene, detect(scene.image, DetectConfig())))
    return summarize(results)


def test_scene_is_reproducible_for_a_seed():
    a, b = generate_scene(seed=3), generate_scene(seed=3)
    assert np.array_equal(a.image, b.image)
    assert not np.array_equal(a.image, generate_scene(seed=4).image)


def test_stars_respect_minimum_separation():
    cfg = SimConfig()
    xy = generate_scene(cfg, seed=1).stars[:, :2]
    dists = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(dists, np.inf)
    assert dists.min() >= cfg.min_star_separation


def test_gaussian_fit_recovers_isolated_star_to_hundredth_of_a_pixel():
    cfg = SimConfig(num_stars=1, num_streaks=0, noise_sigma=0.0)
    scene = generate_scene(cfg, seed=0)
    x, y, _ = scene.stars[0]
    fit = fit_gaussian(scene.image, round(x), round(y))
    assert np.hypot(fit.x - x, fit.y - y) < 0.01
    assert abs(fit.sigma_x - cfg.psf_sigma) < 0.05
    assert abs(fit.sigma_y - cfg.psf_sigma) < 0.05


def test_detects_most_stars_with_few_false_positives():
    summary = run_trials()
    assert summary["star_recall"] > 0.85
    assert summary["star_precision"] > 0.95


def test_subpixel_beats_binary_pixel_centroid():
    rmse = run_trials()["rmse_px"]
    assert rmse["subpixel"] < rmse["binary"]
    assert rmse["subpixel"] < 0.1


def test_streaks_are_classified_separately_from_stars():
    summary = run_trials()
    assert summary["streak_recall"] > 0.8
    assert summary["streak_precision"] > 0.9


def test_streaks_have_wider_fitted_gaussians_than_stars():
    summary = run_trials()
    assert min(summary["streak_sigma_mean"]) > 2 * max(summary["star_sigma_mean"])


def test_no_streaks_means_no_streak_detections():
    scene = generate_scene(SimConfig(num_streaks=0), seed=0)
    detections = detect(scene.image)
    assert not any(d.is_streak for d in detections)


def test_match_points_is_one_to_one_and_respects_max_distance():
    pairs = match_points([(0, 0), (0.5, 0)], [(0, 0)], max_dist=3.0)
    assert len(pairs) == 1
    assert match_points([(10, 10)], [(0, 0)], max_dist=3.0) == []
