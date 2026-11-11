"""Run the full pipeline: simulate -> detect -> localize -> evaluate -> plot.

Examples:
    python -m src.run_pipeline                       # 20 trials, saves a figure
    python -m src.run_pipeline --trials 50 --noise 8
    python -m src.run_pipeline --show                # open the figure window
    python -m src.run_pipeline --save-data           # also write image + labels
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

from .detect import DetectConfig, detect
from .evaluate import METHODS, evaluate_trial, summarize
from .simulate import SimConfig, generate_scene

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0, help="First random seed (trial i uses seed+i).")
    p.add_argument("--trials", type=int, default=20, help="Number of random scenes to evaluate.")
    p.add_argument("--stars", type=int, default=SimConfig.num_stars, help="Stars per image.")
    p.add_argument("--streaks", type=int, default=SimConfig.num_streaks, help="Streaks per image.")
    p.add_argument("--noise", type=float, default=SimConfig.noise_sigma, help="Gaussian noise sigma (counts).")
    p.add_argument("--threshold", type=float, default=DetectConfig.threshold_sigma,
                   help="Detection threshold in noise sigmas above background.")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs", help="Where the figure and metrics go.")
    p.add_argument("--show", action="store_true", help="Display the figure interactively.")
    p.add_argument("--no-plot", action="store_true", help="Skip the figure.")
    p.add_argument("--save-data", action="store_true",
                   help="Write the first scene (image .npy + label CSVs) to data/synthetic/.")
    return p.parse_args()


def save_scene(scene, seed):
    image_dir = ROOT / "data" / "synthetic" / "images"
    label_dir = ROOT / "data" / "synthetic" / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    stem = f"scene_seed{seed}"
    np.save(image_dir / f"{stem}.npy", scene.image)
    np.savetxt(label_dir / f"{stem}_stars.csv", scene.stars, delimiter=",",
               header="x,y,peak", comments="", fmt="%.4f")
    np.savetxt(label_dir / f"{stem}_streaks.csv", scene.streaks, delimiter=",",
               header="x1,y1,x2,y2,peak", comments="", fmt="%.4f")
    print(f"Saved scene and labels for seed {seed} under data/synthetic/")


def plot_example(scene, detections, result, path, show):
    import matplotlib.pyplot as plt

    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1.4, 1]})
    vmax = np.percentile(scene.image, 99.7)
    ax.imshow(scene.image, cmap="gray", vmin=np.median(scene.image), vmax=vmax, origin="upper")
    ax.scatter(scene.stars[:, 0], scene.stars[:, 1], c="red", marker="+", s=60, label="Ground truth star")
    stars = [d for d in detections if not d.is_streak]
    if stars:
        bx, by = zip(*[d.centroid_binary for d in stars])
        ax.scatter(bx, by, facecolors="none", edgecolors="yellow", s=70, label="Binary pixel centroid")
    sub = [d.centroid_subpixel for d in stars if d.fit]
    if sub:
        sx, sy = zip(*sub)
        ax.scatter(sx, sy, c="lime", s=12, label="Sub-pixel Gaussian centroid")
    for i, (x1, y1, x2, y2, _) in enumerate(scene.streaks):
        ax.plot([x1, x2], [y1, y2], c="cyan", lw=0.8, label="Ground truth streak" if i == 0 else None)
    streak_dets = [d.centroid_weighted for d in detections if d.is_streak]
    if streak_dets:
        qx, qy = zip(*streak_dets)
        ax.scatter(qx, qy, c="magenta", marker="x", s=45, label="Detected streak (elongated blob)")
    ax.set_xlim(0, scene.image.shape[1] - 1)
    ax.set_ylim(scene.image.shape[0] - 1, 0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=8, frameon=False)
    ax.set_title("Star and streak detection")

    # Zoom on the matched star with the largest binary-centroid error
    if result.matches:
        worst = max(result.matches, key=lambda m: np.hypot(
            m[0].centroid_binary[0] - scene.stars[m[1], 0], m[0].centroid_binary[1] - scene.stars[m[1], 1]))
        det, ti = worst
        tx, ty = scene.stars[ti, :2]
        r = 6
        x0, y0 = int(round(tx)), int(round(ty))
        zoom.imshow(scene.image, cmap="gray", vmin=np.median(scene.image), vmax=vmax,
                    extent=(-0.5, scene.image.shape[1] - 0.5, scene.image.shape[0] - 0.5, -0.5))
        zoom.scatter([tx], [ty], c="red", marker="+", s=200, label="Ground truth")
        zoom.scatter([det.centroid_binary[0]], [det.centroid_binary[1]], facecolors="none",
                     edgecolors="yellow", s=160, label="Binary centroid")
        zoom.scatter([det.centroid_weighted[0]], [det.centroid_weighted[1]], c="orange", marker="s", s=40,
                     label="Intensity-weighted")
        if det.fit:
            zoom.scatter([det.fit.x], [det.fit.y], c="lime", s=50, label="Gaussian fit")
        zoom.set_xlim(x0 - r - 0.5, x0 + r + 0.5)
        zoom.set_ylim(y0 + r + 0.5, y0 - r - 0.5)
        zoom.legend(fontsize=7, loc="upper right", framealpha=0.85)
        zoom.set_title("Zoom: star with the largest binary-centroid error")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"Saved figure to {path}")
    if show:
        plt.show()
    plt.close(fig)


def print_summary(summary):
    rmse = summary["rmse_px"]
    print(f"\n=== Results over {summary['trials']} scenes ===")
    print(f"Star recall      : {summary['star_recall']:.1%}")
    print(f"Star precision   : {summary['star_precision']:.1%}")
    print(f"Streak recall    : {summary['streak_recall']:.1%}")
    print(f"Streak precision : {summary['streak_precision']:.1%}")
    print("Localization error (RMSE, pixels, matched stars):")
    print(f"  binary centroid      : {rmse['binary']:.3f}")
    print(f"  intensity-weighted   : {rmse['weighted']:.3f}")
    print(f"  Gaussian sub-pixel   : {rmse['subpixel']:.3f}")
    if summary["star_sigma_mean"] and summary["streak_sigma_mean"]:
        s, k = summary["star_sigma_mean"], summary["streak_sigma_mean"]
        print("Mean fitted Gaussian width (sigma_x, sigma_y):")
        print(f"  stars   : ({s[0]:.2f}, {s[1]:.2f})")
        print(f"  streaks : ({k[0]:.2f}, {k[1]:.2f})")


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")

    sim_cfg = replace(SimConfig(), num_stars=args.stars, num_streaks=args.streaks, noise_sigma=args.noise)
    det_cfg = replace(DetectConfig(), threshold_sigma=args.threshold)

    results, first = [], None
    for i in range(args.trials):
        scene = generate_scene(sim_cfg, seed=args.seed + i)
        detections = detect(scene.image, det_cfg)
        result = evaluate_trial(scene, detections)
        results.append(result)
        if first is None:
            first = (scene, detections, result)

    summary = summarize(results)
    print_summary(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved metrics to {metrics_path}")

    if not args.no_plot:
        plot_example(*first, args.out_dir / "detection_example.png", args.show)
    if args.save_data:
        save_scene(first[0], args.seed)


if __name__ == "__main__":
    main()
