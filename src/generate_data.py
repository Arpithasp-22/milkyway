from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from scipy.optimize import curve_fit
import cv2

# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "synthetic" / "images"
LABEL_DIR = PROJECT_ROOT / "data" / "synthetic" / "labels"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Parameters
# =========================
IMAGE_SIZE = 256

NUM_STARS = 50
STAR_INTENSITY = (1200, 2000)
PSF_SIGMA = 1.2

NUM_STREAKS = 5
STREAK_INTENSITY = (1500, 2500)
STREAK_LENGTH = (20, 60)

NOISE_LEVEL = 3
PATCH_RADIUS = 5  # for Gaussian fitting

# =========================
# Star generation
# =========================
def generate_star_field(image):
    stars = []

    for _ in range(NUM_STARS):
        x = np.random.randint(10, IMAGE_SIZE - 10)
        y = np.random.randint(10, IMAGE_SIZE - 10)
        intensity = np.random.uniform(*STAR_INTENSITY)

        image[y, x] += intensity
        stars.append((x, y))

    image = gaussian_filter(image, sigma=PSF_SIGMA)
    return image, stars


# =========================
# Streak generation
# =========================
def add_streaks(image):
    streaks = []

    for _ in range(NUM_STREAKS):
        x1 = np.random.randint(0, IMAGE_SIZE)
        y1 = np.random.randint(0, IMAGE_SIZE)

        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.randint(*STREAK_LENGTH)

        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))

        intensity = np.random.uniform(*STREAK_INTENSITY)

        cv2.line(image, (x1, y1), (x2, y2), int(intensity), 1)
        streaks.append((x1, y1, x2, y2))

    image = gaussian_filter(image, sigma=0.8)
    return image, streaks


# =========================
# Classical blob detection
# =========================
def detect_star_blobs(image_norm, threshold=0.15, min_area=3):
    binary = image_norm > threshold
    labeled, num_objects = ndimage.label(binary)

    centroids = []
    for label in range(1, num_objects + 1):
        mask = labeled == label
        if np.sum(mask) < min_area:
            continue
        y, x = np.where(mask)
        centroids.append((x.mean(), y.mean()))

    return centroids


# =========================
# Gaussian model
# =========================
def gaussian_2d(coords, A, x0, y0, sx, sy, c):
    x, y = coords
    return (
        A * np.exp(-(((x - x0) ** 2) / (2 * sx ** 2)
        + ((y - y0) ** 2) / (2 * sy ** 2))) + c
    ).ravel()


def fit_subpixel_gaussian(image, cx, cy):
    x0, y0 = int(cx), int(cy)

    x_min = max(0, x0 - PATCH_RADIUS)
    x_max = min(IMAGE_SIZE, x0 + PATCH_RADIUS + 1)
    y_min = max(0, y0 - PATCH_RADIUS)
    y_max = min(IMAGE_SIZE, y0 + PATCH_RADIUS + 1)

    patch = image[y_min:y_max, x_min:x_max]
    if patch.size == 0:
        return None

    y, x = np.mgrid[y_min:y_max, x_min:x_max]

    p0 = (
        patch.max(),
        cx,
        cy,
        1.0,
        1.0,
        np.median(patch),
    )

    try:
        popt, _ = curve_fit(
            gaussian_2d,
            (x, y),
            patch.ravel(),
            p0=p0,
            maxfev=2000,
        )
        _, x_fit, y_fit, _, _, _ = popt
        return x_fit, y_fit
    except RuntimeError:
        return None


def normalize(image):
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


# =========================
# Main
# =========================
if __name__ == "__main__":
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    image, gt_stars = generate_star_field(image)
    image, streaks = add_streaks(image)

    image += np.random.normal(0, NOISE_LEVEL, image.shape)
    image = np.clip(image, 0, None)
    image_norm = normalize(image)

    detected = detect_star_blobs(image_norm)

    subpixel_centroids = []
    for cx, cy in detected:
        result = fit_subpixel_gaussian(image, cx, cy)
        if result:
            subpixel_centroids.append(result)

    # =========================
    # Visualization
    # =========================
    plt.figure(figsize=(6, 6))
    plt.imshow(image_norm, cmap="gray")

    gx, gy = zip(*gt_stars)
    plt.scatter(gx, gy, c="red", marker="+", label="Ground Truth")

    if detected:
        dx, dy = zip(*detected)
        plt.scatter(dx, dy, facecolors="none", edgecolors="yellow",
                    s=40, label="Pixel Centroid")

    if subpixel_centroids:
        sx, sy = zip(*subpixel_centroids)
        plt.scatter(sx, sy, c="lime", s=15, label="Sub-pixel Centroid")

    for x1, y1, x2, y2 in streaks:
        plt.plot([x1, x2], [y1, y2], c="cyan")

    plt.legend()
    plt.title("Sub-pixel Star Localization via Gaussian Fitting")
    plt.tight_layout()
    plt.show()

    print(f"Detected stars: {len(detected)}")
    print(f"Sub-pixel fits successful: {len(subpixel_centroids)}")
