"""
EDA Script: Brain Tumor Detection Dataset.

Dataset : brain-tumor-mri (v2, December 2024)
Source  : https://universe.roboflow.com/eksperiment/brain-tumor-mri-ycidy/dataset/2
License : CC BY 4.0

Pre-processing applied by Roboflow:
  - Auto-orientation of pixel data (EXIF-orientation stripping)
  - Resize to 640x640 (Stretch)
  - No augmentation applied

Expected dataset structure:
    data/
        train/   images/  labels/
        valid/   images/  labels/
        test/    images/  labels/
        data.yaml
        README.dataset.txt
"""

import os
import cv2
import yaml
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


#  CLI
def parse_args():
    parser = argparse.ArgumentParser(description="Brain Tumor Detection EDA")
    parser.add_argument("--data", default="./data",
                        help="Path to dataset root (default: ./data)")
    parser.add_argument("--intensity-sample", type=int, default=500,
                        help="Max images sampled for pixel intensity stats (default: 500)")
    return parser.parse_args()


#  CONFIGURATION
SPLITS           = ["train", "valid", "test"]
FALLBACK_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Colors aligned to FALLBACK_CLASSES order:
#   Glioma=#E63946  Meningioma=#457B9D  No Tumor=#2A9D8F  Pituitary=#E9C46A
FALLBACK_COLORS  = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]

def load_class_names(dataset_root):
    """Read class names from data.yaml; fall back to FALLBACK_CLASSES."""
    yaml_path = Path(dataset_root) / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names")
        if names:
            print(f"[INFO] Class names loaded from data.yaml : {names}")
            return names
    print(f"[INFO] data.yaml not found. Using fallback  : {FALLBACK_CLASSES}")
    return FALLBACK_CLASSES


def build_class_colors(class_names):
    """
    Assign colors to classes by name so the mapping survives any class ordering.
    Unknown class names receive a grey fallback.
    """
    name_to_color = {
        "glioma":    "#E63946",
        "meningioma":"#457B9D",
        "no tumor":  "#2A9D8F",
        "no_tumor":  "#2A9D8F",
        "pituitary": "#E9C46A",
    }
    grey = "#AAAAAA"
    return [name_to_color.get(n.lower(), grey) for n in class_names]


def get_no_tumor_idx(class_names):
    """
    Detect the No Tumor class index dynamically.
    Matches any name whose normalised form contains both 'no' and 'tumor'.
    Returns None if not found — caller must handle this case.
    """
    for i, name in enumerate(class_names):
        tokens = name.lower().replace("_", " ").replace("-", " ").split()
        if "no" in tokens and "tumor" in tokens:
            return i
    return None


#  HELPER FUNCTIONS
def get_image_paths(dataset_root, split):
    img_dir = Path(dataset_root) / split / "images"
    return (
        list(img_dir.glob("*.jpg")) +
        list(img_dir.glob("*.png")) +
        list(img_dir.glob("*.jpeg"))
    )


def get_label_path(dataset_root, img_path, split):
    return Path(dataset_root) / split / "labels" / (img_path.stem + ".txt")


def parse_label(label_path):
    """
    Supports both:
      - YOLO bbox format        : class cx cy w h        (5 values)
      - YOLO segmentation format: class x1 y1 x2 y2 ...  (>= 6 coords, even count)
    Segmentation polygons are converted to bounding boxes via min/max of coords.
    Returns list of (class_id, cx, cy, w, h) — all normalized [0, 1].
    """
    annotations = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls    = int(parts[0])
                coords = list(map(float, parts[1:]))

                if len(coords) == 4:
                    cx, cy, w, h = coords
                else:
                    # Drop trailing value if polygon has odd coordinate count
                    if len(coords) % 2 != 0:
                        coords = coords[:-1]
                    xs           = coords[0::2]
                    ys           = coords[1::2]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    w  = x_max - x_min
                    h  = y_max - y_min
                    cx = x_min + w / 2
                    cy = y_min + h / 2

                annotations.append((cls, cx, cy, w, h))
    return annotations


def collect_all_data(dataset_root, class_names, no_tumor_idx, intensity_sample=500):
    """
    Collect image metadata and annotations across all splits.
      - Dimensions are read via PIL (fast, header-only decode).
      - Pixel mean/std computed on a capped random sample for performance.
      - is_no_tumor flag: label file absent OR label file present but empty.
    """
    records = []
    for split in SPLITS:
        for img_path in get_image_paths(dataset_root, split):
            label_path  = get_label_path(dataset_root, img_path, split)
            annotations = parse_label(label_path)

            is_no_tumor = (not label_path.exists()) or (
                label_path.exists() and len(annotations) == 0
            )

            try:
                with Image.open(img_path) as im:
                    w, h       = im.size
                    n_channels = len(im.getbands())
            except Exception:
                continue

            records.append({
                "split":       split,
                "path":        img_path,
                "width":       w,
                "height":      h,
                "channels":    n_channels,
                "annotations": annotations,
                "n_objects":   len(annotations),
                "has_tumor":   not is_no_tumor,
                "is_no_tumor": is_no_tumor,
                "pixel_mean":  None,
                "pixel_std":   None,
            })

    # Pixel stats on random sample only
    sample_indices = set(
        random.sample(range(len(records)), min(intensity_sample, len(records)))
    )
    for i in sample_indices:
        img = cv2.imread(str(records[i]["path"]))
        if img is not None:
            records[i]["pixel_mean"] = float(img.mean())
            records[i]["pixel_std"]  = float(img.std())

    return records


#  SECTION 1: DATASET OVERVIEW
def section1_overview(records, dataset_root, class_names):
    print("=" * 60)
    print("SECTION 1 — DATASET OVERVIEW")
    print("=" * 60)

    total = len(records)
    print(f"Total images found : {total}")
    for split in SPLITS:
        count = sum(1 for r in records if r["split"] == split)
        print(f"  {split:>5} split     : {count} images")

    no_file   = sum(1 for r in records
                    if not get_label_path(dataset_root, r["path"], r["split"]).exists())
    empty     = sum(1 for r in records
                    if r["is_no_tumor"]
                    and get_label_path(dataset_root, r["path"], r["split"]).exists())
    annotated = sum(1 for r in records if r["annotations"])
    multi     = sum(1 for r in records if r["n_objects"] > 1)

    print(f"\nLabel coverage:")
    print(f"  Annotated images            : {annotated}")
    print(f"  Empty label file (no tumor) : {empty}")
    print(f"  Missing label file          : {no_file}")
    print(f"  Images with >1 annotation   : {multi}")

    all_anns = [a for r in records for a in r["annotations"]]
    print(f"\nTotal bounding boxes : {len(all_anns)}")
    print(f"Classes              : {class_names}")
    print(f"Dataset source       : https://universe.roboflow.com/eksperiment/brain-tumor-mri-ycidy/dataset/2")
    print(f"License              : CC BY 4.0")


#  SECTION 2: CLASS DISTRIBUTION
def section2_class_distribution(records, class_names, class_colors, no_tumor_idx):
    print("\n" + "=" * 60)
    print("SECTION 2 — CLASS DISTRIBUTION")
    print("=" * 60)

    overall_counts = defaultdict(int)
    split_counts   = {s: defaultdict(int) for s in SPLITS}

    for r in records:
        for (cls, *_) in r["annotations"]:
            overall_counts[cls] += 1
            split_counts[r["split"]][cls] += 1
        # Count No Tumor using the dynamically resolved index
        if r["is_no_tumor"] and no_tumor_idx is not None:
            overall_counts[no_tumor_idx] += 1
            split_counts[r["split"]][no_tumor_idx] += 1

    print("\nOverall class counts:")
    for i, name in enumerate(class_names):
        print(f"  {name:15}: {overall_counts[i]}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Class Distribution Analysis", fontsize=14, fontweight="bold")

    # Plot 1: Overall bar chart
    ax     = axes[0]
    counts = [overall_counts[i] for i in range(len(class_names))]
    bars   = ax.bar(class_names, counts, color=class_colors,
                    edgecolor="black", linewidth=0.7)
    ax.set_title("Overall Sample Count per Class")
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=10)

    # Plot 2: Per-split grouped bar
    ax2          = axes[1]
    x            = np.arange(len(class_names))
    width        = 0.25
    split_colors = ["#4361EE", "#F77F00", "#4CC9F0"]

    for i, split in enumerate(SPLITS):
        vals = [split_counts[split][j] for j in range(len(class_names))]
        ax2.bar(x + i * width, vals, width, label=split.capitalize(),
                color=split_colors[i], edgecolor="black", linewidth=0.5)

    ax2.set_xticks(x + width)
    ax2.set_xticklabels(class_names)
    ax2.set_title("Class Distribution per Split")
    ax2.set_ylabel("Count")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("eda_figs/eda_class_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()


#  SECTION 3: IMAGE PROPERTIES
def section3_image_properties(records):
    print("\n" + "=" * 60)
    print("SECTION 3 — IMAGE PROPERTIES")
    print("=" * 60)
    print("  Roboflow pre-processing resized all images to 640x640.")
    channels = [r["channels"] for r in records]
    print(f"Channels — unique values: {sorted(set(channels))}")


#  SECTION 4: SAMPLE VISUALIZATION
def section4_sample_visualization(records, class_names, class_colors, no_tumor_idx):
    print("\n" + "=" * 60)
    print("SECTION 4 — SAMPLE VISUALIZATION PER CLASS")
    print("=" * 60)

    samples = {}

    for r in records:
        for (cls, cx, cy, w, h) in r["annotations"]:
            if cls not in samples:
                samples[cls] = (r["path"], r["width"], r["height"], cx, cy, w, h)
        if len(samples) == len(class_names):
            break

    # No Tumor images have empty label files — collect separately
    if no_tumor_idx is not None and no_tumor_idx not in samples:
        for r in records:
            if r["is_no_tumor"]:
                samples[no_tumor_idx] = (r["path"], r["width"], r["height"],
                                         None, None, None, None)
                break

    fig, axes = plt.subplots(1, len(class_names), figsize=(16, 4))
    fig.suptitle("Sample MRI Image per Class (with Bounding Box where applicable)",
                 fontsize=13, fontweight="bold")

    for cls_id, ax in enumerate(axes):
        if cls_id not in samples:
            ax.axis("off")
            ax.set_title(f"{class_names[cls_id]}\n(No sample found)")
            continue

        path, img_w, img_h, cx, cy, bw, bh = samples[cls_id]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)

        if cx is not None:
            x1    = int((cx - bw / 2) * img_w)
            y1    = int((cy - bh / 2) * img_h)
            box_w = int(bw * img_w)
            box_h = int(bh * img_h)
            rect  = patches.Rectangle(
                (x1, y1), box_w, box_h,
                linewidth=2, edgecolor=class_colors[cls_id], facecolor="none"
            )
            ax.add_patch(rect)

        ax.set_title(class_names[cls_id], fontsize=11, fontweight="bold",
                     color=class_colors[cls_id])
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("eda_figs/eda_samples_per_class.png", dpi=300, bbox_inches="tight")
    plt.show()


#  SECTION 5: BOUNDING BOX ANALYSIS
def _kmeans_anchors(box_wh, k=9, iters=300):
    """
    IoU-based k-means anchor clustering.
    box_wh : np.ndarray of shape (N, 2) — normalized [w, h] pairs.
    Returns anchors sorted by area, shape (k, 2). Returns None if N < k.
    """
    if len(box_wh) < k:
        return None

    np.random.seed(42)
    anchors = box_wh[np.random.choice(len(box_wh), k, replace=False)]

    for _ in range(iters):
        inter   = (np.minimum(box_wh[:, None, 0], anchors[None, :, 0]) *
                   np.minimum(box_wh[:, None, 1], anchors[None, :, 1]))
        union   = (box_wh[:, None, 0] * box_wh[:, None, 1] +
                   anchors[None, :, 0] * anchors[None, :, 1] - inter)
        iou     = inter / (union + 1e-9)
        cluster = np.argmax(iou, axis=1)

        new_anchors = np.array([
            box_wh[cluster == j].mean(axis=0) if (cluster == j).any() else anchors[j]
            for j in range(k)
        ])
        if np.allclose(new_anchors, anchors, atol=1e-6):
            break
        anchors = new_anchors

    areas = anchors[:, 0] * anchors[:, 1]
    return anchors[np.argsort(areas)]


def section5_bounding_box_analysis(records, class_names, class_colors):
    print("\n" + "=" * 60)
    print("SECTION 5 — BOUNDING BOX ANALYSIS (Localization)")
    print("=" * 60)

    box_data = defaultdict(lambda: {
        "widths": [], "heights": [], "cx": [], "cy": [],
        "areas": [], "aspects": []
    })

    for r in records:
        for (cls, cx, cy, bw, bh) in r["annotations"]:
            box_data[cls]["widths"].append(bw)
            box_data[cls]["heights"].append(bh)
            box_data[cls]["cx"].append(cx)
            box_data[cls]["cy"].append(cy)
            box_data[cls]["areas"].append(bw * bh)
            box_data[cls]["aspects"].append(bw / bh if bh > 0 else 0)

    for cls_id, name in enumerate(class_names):
        if not box_data[cls_id]["areas"]:
            continue
        areas   = box_data[cls_id]["areas"]
        aspects = box_data[cls_id]["aspects"]
        print(f"\n  {name}:")
        print(f"    Box count     : {len(areas)}")
        print(f"    Area   (norm) — mean: {np.mean(areas):.4f}, "
              f"min: {min(areas):.4f}, max: {max(areas):.4f}")
        print(f"    Width  (norm) — mean: {np.mean(box_data[cls_id]['widths']):.4f}")
        print(f"    Height (norm) — mean: {np.mean(box_data[cls_id]['heights']):.4f}")
        print(f"    Aspect (w/h)  — mean: {np.mean(aspects):.4f}, "
              f"min: {min(aspects):.4f}, max: {max(aspects):.4f}")

    all_wh = np.array([
        [bw, bh]
        for r in records
        for (_, _, _, bw, bh) in r["annotations"]
    ])

    k = 9
    print(f"\n  K-means anchor suggestion (k={k}, normalized):")
    anchors = _kmeans_anchors(all_wh, k=k)
    if anchors is not None:
        for i, (w, h) in enumerate(anchors):
            print(f"    Anchor {i+1}: w={w:.4f}, h={h:.4f}  (area≈{w*h:.4f})")
    else:
        print(f"    Not enough boxes to compute {k} anchors.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Bounding Box Analysis per Class", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    for cls_id, name in enumerate(class_names):
        if box_data[cls_id]["areas"]:
            ax.hist(box_data[cls_id]["areas"], bins=20, alpha=0.6, label=name,
                    color=class_colors[cls_id], edgecolor="black", linewidth=0.3)
    ax.set_title("Bounding Box Area Distribution (normalized)")
    ax.set_xlabel("Area (w × h)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    ax2 = axes[0, 1]
    for cls_id, name in enumerate(class_names):
        if box_data[cls_id]["widths"]:
            ax2.scatter(box_data[cls_id]["widths"], box_data[cls_id]["heights"],
                        alpha=0.4, s=10, label=name, color=class_colors[cls_id])
    ax2.set_title("Box Width vs Height (normalized)")
    ax2.set_xlabel("Width")
    ax2.set_ylabel("Height")
    ax2.legend(fontsize=8)

    ax3 = axes[1, 0]
    for cls_id, name in enumerate(class_names):
        if box_data[cls_id]["aspects"]:
            ax3.hist(box_data[cls_id]["aspects"], bins=20, alpha=0.6, label=name,
                     color=class_colors[cls_id], edgecolor="black", linewidth=0.3)
    ax3.set_title("Bounding Box Aspect Ratio Distribution (w/h)")
    ax3.set_xlabel("Aspect Ratio")
    ax3.set_ylabel("Count")
    ax3.legend(fontsize=8)

    ax4    = axes[1, 1]
    all_cx = [cx for cls_id in box_data for cx in box_data[cls_id]["cx"]]
    all_cy = [cy for cls_id in box_data for cy in box_data[cls_id]["cy"]]
    h2d, _, _ = np.histogram2d(all_cx, all_cy, bins=20, range=[[0, 1], [0, 1]])
    im = ax4.imshow(h2d.T, origin="lower", cmap="hot",
                    extent=[0, 1, 0, 1], aspect="auto")
    plt.colorbar(im, ax=ax4, label="Count")
    ax4.set_title("Bounding Box Center Heatmap (all classes)")
    ax4.set_xlabel("Center X (normalized)")
    ax4.set_ylabel("Center Y (normalized)")

    plt.tight_layout()
    plt.savefig("eda_figs/eda_bbox_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


#  SECTION 6: SPLIT SUMMARY
def section6_split_summary(records, class_names, class_colors, no_tumor_idx):
    print("\n" + "=" * 60)
    print("SECTION 6 — TRAIN / VAL / TEST SPLIT SUMMARY")
    print("=" * 60)

    split_totals = defaultdict(int)
    for r in records:
        split_totals[r["split"]] += 1

    total_imgs = len(records)
    print(f"\n{'Split':<10} {'Images':>8} {'% of Total':>12}")
    print("-" * 32)
    for split in SPLITS:
        n   = split_totals[split]
        pct = 100 * n / total_imgs if total_imgs else 0
        print(f"{split:<10} {n:>8} {pct:>11.1f}%")

    split_class_counts = {s: defaultdict(int) for s in SPLITS}
    for r in records:
        for (cls, *_) in r["annotations"]:
            split_class_counts[r["split"]][cls] += 1
        if r["is_no_tumor"] and no_tumor_idx is not None:
            split_class_counts[r["split"]][no_tumor_idx] += 1

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Class Distribution per Split (Proportion)",
                 fontsize=13, fontweight="bold")

    for i, split in enumerate(SPLITS):
        counts = [split_class_counts[split][j] for j in range(len(class_names))]
        if sum(counts) == 0:
            axes[i].axis("off")
            continue
        axes[i].pie(counts, labels=class_names, colors=class_colors,
                    autopct="%1.1f%%", startangle=140,
                    textprops={"fontsize": 8})
        axes[i].set_title(f"{split.capitalize()} Split\n({split_totals[split]} images)")

    plt.tight_layout()
    plt.savefig("eda_figs/eda_split_summary.png", dpi=300, bbox_inches="tight")
    plt.show()


#  MAIN
if __name__ == "__main__":
    args         = parse_args()
    DATASET_ROOT = args.data

    print("\n" + "-" * 60)
    print("  BRAIN TUMOR DETECTION — EXPLORATORY DATA ANALYSIS")
    print("-" * 60)
    print(f"\nDataset root : {os.path.abspath(DATASET_ROOT)}")

    class_names  = load_class_names(DATASET_ROOT)
    class_colors = build_class_colors(class_names)
    no_tumor_idx = get_no_tumor_idx(class_names)

    print(f"Loading dataset")
    records = collect_all_data(DATASET_ROOT, class_names, no_tumor_idx,
                               args.intensity_sample)

    if not records:
        print("\n[ERROR] No images found. Please check the dataset root path.")
        print(f"  Expected: {DATASET_ROOT}/train/images/  and  {DATASET_ROOT}/train/labels/")
        exit(1)

    print(f"Loaded {len(records)} images successfully.\n")

    section1_overview(records, DATASET_ROOT, class_names)
    section2_class_distribution(records, class_names, class_colors, no_tumor_idx)
    section3_image_properties(records)
    section4_sample_visualization(records, class_names, class_colors, no_tumor_idx)
    section5_bounding_box_analysis(records, class_names, class_colors)
    section6_split_summary(records, class_names, class_colors, no_tumor_idx)

    print("\n" + "-" * 60)
    print("EDA COMPLETE")
    print("-" * 60)
