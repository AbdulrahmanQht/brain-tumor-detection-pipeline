"""
EDA Script: Brain Tumor Detection Dataset.
Dataset 1: Medical Image DataSet - Brain Tumor Detection (YOLOv11 split)
Kaggle: https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection

Dataset 2: Br35H :: Brain Tumor Detection 2020
Kaggle: https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection

Expected dataset structure:
    data/
        train/
            images/  labels/
        valid/
            images/  labels/
        test/
            images/  labels/
        data.yaml
"""
import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# CONFIGURATION
DATASET_ROOT = "./data"
SPLITS       = ["train", "valid", "test"]
CLASS_NAMES  = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
CLASS_COLORS = ["#E63946", "#457B9D", "#E9C46A", "#2A9D8F"]

# HELPER FUNCTIONS
def get_image_paths(split):
    img_dir = Path(DATASET_ROOT) / split / "images"
    return list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))

def get_label_path(img_path, split):
    return Path(DATASET_ROOT) / split / "labels" / (img_path.stem + ".txt")

def parse_label(label_path):
    """
    Supports both:
      - YOLO bbox format:       class cx cy w h  (5 values)
      - YOLO segmentation format: class x1 y1 x2 y2 ... (odd count >= 7)
    Segmentation polygons are converted to bounding boxes via min/max of coords.
    Returns list of (class_id, cx, cy, w, h) — all normalized.
    """
    annotations = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                coords = list(map(float, parts[1:]))
                if len(coords) == 4:
                    # Standard bbox: cx cy w h
                    cx, cy, w, h = coords
                else:
                    # Segmentation polygon: x1 y1 x2 y2 ...
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    w  = x_max - x_min
                    h  = y_max - y_min
                    cx = x_min + w / 2
                    cy = y_min + h / 2
                annotations.append((cls, cx, cy, w, h))
    return annotations

def collect_all_data():
    """Collect image metadata and annotations across all splits."""
    records = []
    for split in SPLITS:
        for img_path in get_image_paths(split):
            label_path = get_label_path(img_path, split)
            annotations = parse_label(label_path)

            # If empty label file → this is a No Tumor image (class 3)
            is_no_tumor = label_path.exists() and len(annotations) == 0

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            records.append({
                "split":       split,
                "path":        img_path,
                "width":       w,
                "height":      h,
                "channels":    img.shape[2] if len(img.shape) == 3 else 1,
                "annotations": annotations,
                "n_objects":   len(annotations),
                "has_tumor":   not is_no_tumor,
                "pixel_mean":  img.mean(),
                "pixel_std":   img.std(),
                "is_no_tumor": is_no_tumor,  # new flag
            })
    return records

# 1. DATASET OVERVIEW
def section1_overview(records):
    print("=" * 60)
    print("SECTION 1 — DATASET OVERVIEW")
    print("=" * 60)
    total = len(records)
    print(f"Total images found : {total}")
    for split in SPLITS:
        count = sum(1 for r in records if r["split"] == split)
        print(f"  {split:>5} split     : {count} images")

    all_anns = [a for r in records for a in r["annotations"]]
    print(f"\nTotal bounding boxes: {len(all_anns)}")
    print(f"Classes             : {CLASS_NAMES}")
    print(f"Dataset 1 source      : https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection")
    print(f"Dataset 2 source      : https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection")

# 2. CLASS DISTRIBUTION ANALYSIS
def section2_class_distribution(records):
    print("\n" + "=" * 60)
    print("SECTION 2 — CLASS DISTRIBUTION")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Class Distribution Analysis", fontsize=14, fontweight="bold")

    # Overall class counts across all annotations
    overall_counts = defaultdict(int)
    split_counts = {s: defaultdict(int) for s in SPLITS}

    for r in records:
        for (cls, *_) in r["annotations"]:
            overall_counts[cls] += 1
            split_counts[r["split"]][cls] += 1
        # Count No Tumor images explicitly
        if r["is_no_tumor"]:
            overall_counts[3] += 1
            split_counts[r["split"]][3] += 1

    # Plot 1: Overall
    ax = axes[0]
    counts = [overall_counts[i] for i in range(len(CLASS_NAMES))]
    bars = ax.bar(CLASS_NAMES, counts, color=CLASS_COLORS, edgecolor="black", linewidth=0.7)
    ax.set_title("Overall Bounding Box Count per Class")
    ax.set_ylabel("Count")
    ax.set_xlabel("Tumor Class")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=10)
    print("\nOverall class counts:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:15}: {overall_counts[i]}")

    # Plot 2: Per-split stacked
    ax2 = axes[1]
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    split_colors = ["#4361EE", "#F77F00", "#4CC9F0"]
    bottoms = np.zeros(len(CLASS_NAMES))
    for i, split in enumerate(SPLITS):
        vals = [split_counts[split][j] for j in range(len(CLASS_NAMES))]
        ax2.bar(x + i * width, vals, width, label=split.capitalize(),
                color=split_colors[i], edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(CLASS_NAMES)
    ax2.set_title("Class Distribution per Split")
    ax2.set_ylabel("Count")
    ax2.legend()

    plt.tight_layout()
    plt.show()

# 3. IMAGE PROPERTY ANALYSIS
def section3_image_properties(records):
    print("\n" + "=" * 60)
    print("SECTION 3 — IMAGE PROPERTIES")
    print("=" * 60)

    widths   = [r["width"]  for r in records]
    heights  = [r["height"] for r in records]
    channels = [r["channels"] for r in records]

    print(f"Width  — min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.1f}")
    print(f"Height — min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.1f}")
    print(f"Channels — unique values: {set(channels)}")

    resolutions = set(zip(widths, heights))
    print(f"Unique resolutions found: {len(resolutions)}")
    if len(resolutions) <= 10:
        for r in sorted(resolutions):
            print(f"  {r[0]}x{r[1]}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Image Resolution Distribution", fontsize=13, fontweight="bold")

    axes[0].hist(widths,  bins=20, color="#4361EE", edgecolor="black", linewidth=0.5)
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Count")

    axes[1].hist(heights, bins=20, color="#F77F00", edgecolor="black", linewidth=0.5)
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Height (px)")

    plt.tight_layout()
    plt.show()

# 4. SAMPLE VISUALIZATION (per class)
def section4_sample_visualization(records):
    print("\n" + "=" * 60)
    print("SECTION 4 — SAMPLE VISUALIZATION PER CLASS")
    print("=" * 60)

    # Pick one representative sample per class
    samples = {}
    for r in records:
        for (cls, cx, cy, w, h) in r["annotations"]:
            if cls not in samples:
                samples[cls] = (r["path"], r["width"], r["height"], cx, cy, w, h)
        if len(samples) == len(CLASS_NAMES):
            break

    # Collect No Tumor samples separately (images with empty labels)
    NO_TUMOR_IDX = 3
    if NO_TUMOR_IDX not in samples:
        for r in records:
            if not r["annotations"] and r["split"] == "train":
                samples[NO_TUMOR_IDX] = (r["path"], r["width"], r["height"], None, None, None, None)
                break

    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(16, 4))
    fig.suptitle("Sample MRI Image per Tumor Class (with Bounding Box)", fontsize=13, fontweight="bold")

    for cls_id, ax in enumerate(axes):
        if cls_id not in samples:
            ax.axis("off")
            ax.set_title(f"{CLASS_NAMES[cls_id]}\n(No sample found)")
            continue

        path, img_w, img_h, cx, cy, bw, bh = samples[cls_id]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)

        if cx is not None:  # only draw box if there's an annotation
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            box_w = int(bw * img_w)
            box_h = int(bh * img_h)
            rect = patches.Rectangle((x1, y1), box_w, box_h,
                                     linewidth=2, edgecolor=CLASS_COLORS[cls_id], facecolor="none")
            ax.add_patch(rect)

        ax.set_title(CLASS_NAMES[cls_id], fontsize=11, fontweight="bold", color=CLASS_COLORS[cls_id])
        ax.axis("off")


# 5. BOUNDING BOX ANALYSIS (localization-specific)
def section5_bounding_box_analysis(records):
    print("\n" + "=" * 60)
    print("SECTION 5 — BOUNDING BOX ANALYSIS (Localization)")
    print("=" * 60)

    box_data = defaultdict(lambda: {"widths": [], "heights": [], "cx": [], "cy": [], "areas": []})

    for r in records:
        for (cls, cx, cy, bw, bh) in r["annotations"]:
            box_data[cls]["widths"].append(bw)
            box_data[cls]["heights"].append(bh)
            box_data[cls]["cx"].append(cx)
            box_data[cls]["cy"].append(cy)
            box_data[cls]["areas"].append(bw * bh)

    # Print stats
    for cls_id, name in enumerate(CLASS_NAMES):
        if not box_data[cls_id]["areas"]:
            continue
        areas = box_data[cls_id]["areas"]
        print(f"\n  {name}:")
        print(f"    Box count : {len(areas)}")
        print(f"    Area (norm) — mean: {np.mean(areas):.4f}, min: {min(areas):.4f}, max: {max(areas):.4f}")
        print(f"    Width (norm)— mean: {np.mean(box_data[cls_id]['widths']):.4f}")
        print(f"    Height(norm)— mean: {np.mean(box_data[cls_id]['heights']):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Bounding Box Analysis per Class", fontsize=13, fontweight="bold")

    # Plot 1: Box area distribution per class
    ax = axes[0]
    for cls_id, name in enumerate(CLASS_NAMES):
        if box_data[cls_id]["areas"]:
            ax.hist(box_data[cls_id]["areas"], bins=20, alpha=0.6,
                    label=name, color=CLASS_COLORS[cls_id], edgecolor="black", linewidth=0.3)
    ax.set_title("Bounding Box Area Distribution\n(normalized)")
    ax.set_xlabel("Area (w × h)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # Plot 2: Box width vs height scatter
    ax2 = axes[1]
    for cls_id, name in enumerate(CLASS_NAMES):
        if box_data[cls_id]["widths"]:
            ax2.scatter(box_data[cls_id]["widths"], box_data[cls_id]["heights"],
                        alpha=0.4, s=10, label=name, color=CLASS_COLORS[cls_id])
    ax2.set_title("Box Width vs Height\n(normalized)")
    ax2.set_xlabel("Width")
    ax2.set_ylabel("Height")
    ax2.legend(fontsize=8)

    # Plot 3: Bounding box center heatmap
    ax3 = axes[2]
    all_cx = [cx for cls_id in box_data for cx in box_data[cls_id]["cx"]]
    all_cy = [cy for cls_id in box_data for cy in box_data[cls_id]["cy"]]
    h2d, xedges, yedges = np.histogram2d(all_cx, all_cy, bins=20, range=[[0,1],[0,1]])
    im = ax3.imshow(h2d.T, origin="lower", cmap="hot", extent=[0, 1, 0, 1], aspect="auto")
    plt.colorbar(im, ax=ax3, label="Count")
    ax3.set_title("Bounding Box Center Heatmap\n(all classes)")
    ax3.set_xlabel("Center X (normalized)")
    ax3.set_ylabel("Center Y (normalized)")

    plt.tight_layout()
    plt.show()

# 6. PIXEL INTENSITY ANALYSIS
def section6_pixel_intensity(records):
    print("\n" + "=" * 60)
    print("SECTION 6 — PIXEL INTENSITY ANALYSIS")
    print("=" * 60)

    # Group images by primary class (first annotation)
    class_means = defaultdict(list)
    class_stds  = defaultdict(list)

    for r in records:
        if r["annotations"]:
            primary_cls = r["annotations"][0][0]
        elif r["is_no_tumor"]:
            primary_cls = 3  # No Tumor
        else:
            continue
        class_means[primary_cls].append(r["pixel_mean"])
        class_stds[primary_cls].append(r["pixel_std"])

    print("\nMean pixel intensity per class:")
    for cls_id, name in enumerate(CLASS_NAMES):
        if class_means[cls_id]:
            print(f"  {name:15}: mean={np.mean(class_means[cls_id]):.2f}, std={np.mean(class_stds[cls_id]):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Pixel Intensity Analysis per Class", fontsize=13, fontweight="bold")

    ax = axes[0]
    for cls_id, name in enumerate(CLASS_NAMES):
        if class_means[cls_id]:
            ax.hist(class_means[cls_id], bins=25, alpha=0.6,
                    label=name, color=CLASS_COLORS[cls_id], edgecolor="black", linewidth=0.3)
    ax.set_title("Distribution of Mean Pixel Intensity")
    ax.set_xlabel("Mean Pixel Value (0–255)")
    ax.set_ylabel("Image Count")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    data_to_plot = [class_means[i] for i in range(len(CLASS_NAMES)) if class_means[i]]
    labels_to_plot = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES)) if class_means[i]]
    bp = ax2.boxplot(data_to_plot, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], [CLASS_COLORS[i] for i in range(len(CLASS_NAMES)) if class_means[i]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_xticklabels(labels_to_plot, rotation=15, fontsize=9)
    ax2.set_title("Pixel Intensity Boxplot per Class")
    ax2.set_ylabel("Mean Pixel Value")

    plt.tight_layout()
    plt.show()

# 7. TRAIN / VAL / TEST SPLIT SUMMARY
def section7_split_summary(records):
    print("\n" + "=" * 60)
    print("SECTION 7 — TRAIN / VAL / TEST SPLIT SUMMARY")
    print("=" * 60)

    split_class_counts = {s: defaultdict(int) for s in SPLITS}
    split_totals = defaultdict(int)

    for r in records:
        split_totals[r["split"]] += 1
        for (cls, *_) in r["annotations"]:
            split_class_counts[r["split"]][cls] += 1
        if r["is_no_tumor"]:
            split_class_counts[r["split"]][3] += 1

    total_imgs = len(records)
    print(f"\n{'Split':<10} {'Images':>8} {'% of Total':>12}")
    print("-" * 32)
    for split in SPLITS:
        n = split_totals[split]
        pct = 100 * n / total_imgs if total_imgs else 0
        print(f"{split:<10} {n:>8} {pct:>11.1f}%")

    print(f"\n{'Class':<16} {'Train':>8} {'Val':>8} {'Test':>8}")
    print("-" * 44)
    for cls_id, name in enumerate(CLASS_NAMES):
        tr = split_class_counts["train"][cls_id]
        va = split_class_counts["valid"][cls_id]
        te = split_class_counts["test"][cls_id]
        print(f"{name:<16} {tr:>8} {va:>8} {te:>8}")

    # Pie charts
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Class Distribution per Split", fontsize=13, fontweight="bold")

    for i, split in enumerate(SPLITS):
        counts = [split_class_counts[split][j] for j in range(len(CLASS_NAMES))]
        if sum(counts) == 0:
            axes[i].axis("off")
            continue
        axes[i].pie(counts, labels=CLASS_NAMES, colors=CLASS_COLORS,
                    autopct="%1.1f%%", startangle=140,
                    textprops={"fontsize": 8})
        axes[i].set_title(f"{split.capitalize()} Split\n({split_totals[split]} images)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("\n" + "-" * 60)
    print("  BRAIN TUMOR DETECTION — EXPLORATORY DATA ANALYSIS")
    print("-" * 60)
    print(f"\nDataset root : {os.path.abspath(DATASET_ROOT)}")

    print("Loading dataset...")
    records = collect_all_data()

    if not records:
        print("\n[ERROR] No images found. Please check DATASET_ROOT path.")
        print(f"  Expected structure: {DATASET_ROOT}/train/images/  and  {DATASET_ROOT}/train/labels/")
        exit(1)

    print(f"Loaded {len(records)} images successfully.\n")

    section1_overview(records)
    section2_class_distribution(records)
    section3_image_properties(records)
    section4_sample_visualization(records)
    section5_bounding_box_analysis(records)
    section6_pixel_intensity(records)
    section7_split_summary(records)

    print("\n" + "-" * 60)
    print("EDA COMPLETE")
    print("-" * 60)