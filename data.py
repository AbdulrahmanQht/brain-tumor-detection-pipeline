from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
YOLO_ID_TO_CLASS = {
    0: "glioma",
    1: "meningioma",
    2: "no_tumor",
    3: "pituitary",
}
YOLO_CLASS_TO_ID = {class_name: class_id for class_id, class_name in YOLO_ID_TO_CLASS.items()}
SPLITS = ("train", "valid", "test")
SPLIT_ALIASES = {"val": "valid", "validation": "valid"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class YoloTarget:
    """Single-image target in normalized YOLO xywh format."""

    boxes: torch.Tensor
    labels: torch.Tensor
    image_id: str
    image_path: str
    label_path: str


class BrainTumorDataset(Dataset):
    """
    Dataset for the original Roboflow YOLO-format MRI scans.

    Returns:
        image: Float tensor shaped [3, H, W], scaled to [0, 1]
        target: dict with normalized YOLO boxes [N, 4] and class labels [N]
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        image_size: int = 640,
        augment: bool | None = None,
        transform: Callable[[Image.Image, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = normalize_split(split)
        self.image_size = image_size
        self.images_dir = self.root_dir / self.split / "images"
        self.labels_dir = self.root_dir / self.split / "labels"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing images directory: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Missing labels directory: {self.labels_dir}")

        self.image_paths = sorted(
            path
            for path in self.images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")

        self.augment = split == "train" if augment is None else augment
        self.transform = transform or self._build_transform()

    def _build_transform(
        self,
    ) -> Callable[[Image.Image, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        def transform(
            image: Image.Image,
            boxes: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            if self.augment:
                image, boxes = _apply_yolo_augmentation(image, boxes)
            return _to_tensor(image), boxes

        return transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image_path = self.image_paths[index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        boxes, labels = read_yolo_label(label_path)
        image_tensor, boxes = self.transform(image, boxes)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_path.stem,
            "image_path": str(image_path),
            "label_path": str(label_path),
        }
        return image_tensor, target


class ROIDataset(Dataset):
    """
    Dataset for split-aligned ROI crops saved as class folders.

    Expected structure:
        roi_dataset/train/glioma/*.jpg
        roi_dataset/train/meningioma/*.jpg
        roi_dataset/train/pituitary/*.jpg
        roi_dataset/train/no_tumor/*.jpg
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        classes: Iterable[str] | None = None,
        roi_size: int = 224,
        augment: bool | None = None,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = normalize_split(split)
        self.classes = list(classes or DEFAULT_CLASSES)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.split_dir = self.root_dir / self.split

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Missing ROI split directory: {self.split_dir}")

        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"No ROI images found in {self.split_dir}")

        self.augment = split == "train" if augment is None else augment
        self.transform = transform or self._build_transform(roi_size)

    def _collect_samples(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        for class_name in self.classes:
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((path, self.class_to_idx[class_name]))
        return samples

    def _build_transform(self, roi_size: int) -> _ROITransform:
        return _ROITransform(roi_size=roi_size, augment=self.augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label
    
class _ROITransform:
    """Picklable transform for ROIDataset — defined at module level for Windows spawn."""

    def __init__(self, roi_size: int, augment: bool) -> None:
        self.roi_size = roi_size
        self.augment = augment

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.roi_size, self.roi_size), Image.BILINEAR)

        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            image = image.rotate(random.uniform(-15.0, 15.0), resample=Image.BILINEAR)
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)

        tensor = _to_tensor(image)

        if self.augment:
            noise = torch.randn_like(tensor) * 0.02
            tensor = (tensor + noise).clamp(0.0, 1.0)

        return (tensor - IMAGENET_MEAN) / IMAGENET_STD
def read_yolo_label(label_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read one YOLO label file.

    Empty files are valid for No Tumor images and return zero boxes.
    Box format is normalized [x_center, y_center, width, height].
    """

    path = Path(label_path)
    if not path.exists() or path.stat().st_size == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

    boxes: list[list[float]] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label at {path}:{line_number}: {line!r}")
            class_id, x_center, y_center, width, height = parts
            labels.append(int(class_id))
            boxes.append([float(x_center), float(y_center), float(width), float(height)])

    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def normalize_split(split: str) -> str:
    normalized = SPLIT_ALIASES.get(split.lower(), split.lower())
    if normalized not in SPLITS:
        raise ValueError(f"Unknown split {split!r}. Expected one of {SPLITS} or val.")
    return normalized


def yolo_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, Any]]]
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)


def load_yolo_dataset(config: dict[str, Any]) -> dict[str, DataLoader]:
    """
    Return train/valid/test DataLoaders for the original YOLO-format dataset.

    The Ultralytics trainer can still consume dataset/data.yaml directly; these
    loaders are useful for custom evaluation, ROI extraction, and sanity checks.
    """

    dataset_path = config.get("dataset_path", "./dataset/")
    image_size = int(config.get("image_size", 640))
    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", torch.cuda.is_available()))

    loaders: dict[str, DataLoader] = {}
    for split in SPLITS:
        dataset = BrainTumorDataset(dataset_path, split=split, image_size=image_size)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=yolo_collate_fn,
        )
    return loaders


def load_roi_dataset(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return train, validation, and test DataLoaders for classifier training."""

    roi_dataset_path = config.get("roi_dataset_path", "./roi_dataset/")
    classes = config.get("classes", DEFAULT_CLASSES)
    roi_size = int(config.get("roi_size", 224))
    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", torch.cuda.is_available()))

    datasets = {
        split: ROIDataset(roi_dataset_path, split=split, classes=classes, roi_size=roi_size)
        for split in SPLITS
    }
    return (
        DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        DataLoader(
            datasets["valid"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )


def compute_class_weights(
    train_dir: str | Path,
    classes: Iterable[str] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from ROI training folders only.

    Formula:
        weight_c = total_samples / (num_classes * samples_in_class_c)
    """

    root = Path(train_dir)
    class_names = list(classes or DEFAULT_CLASSES)
    counts = torch.tensor(
        [_count_images(root / class_name) for class_name in class_names],
        dtype=torch.float32,
    )

    if torch.any(counts == 0):
        missing = [class_names[idx] for idx, count in enumerate(counts.tolist()) if count == 0]
        raise ValueError(f"Cannot compute class weights with empty classes: {missing}")

    weights = counts.sum() / (len(class_names) * counts)
    if device is not None:
        weights = weights.to(device)
    return weights


def _count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _apply_yolo_augmentation(
    image: Image.Image,
    boxes: torch.Tensor,
) -> tuple[Image.Image, torch.Tensor]:
    width, height = image.size
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = _flip_boxes_horizontal(boxes)
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        boxes = _flip_boxes_vertical(boxes)

    angle = random.uniform(-15.0, 15.0)
    image = image.rotate(angle, resample=Image.BILINEAR)
    boxes = _rotate_boxes(boxes, angle, width, height)

    brightness = random.uniform(0.8, 1.2)
    contrast = random.uniform(0.8, 1.2)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return image, boxes


def _flip_boxes_horizontal(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    flipped = boxes.clone()
    flipped[:, 0] = 1.0 - flipped[:, 0]
    return flipped


def _flip_boxes_vertical(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    flipped = boxes.clone()
    flipped[:, 1] = 1.0 - flipped[:, 1]
    return flipped


def _rotate_boxes(
    boxes: torch.Tensor,
    angle_degrees: float,
    width: int,
    height: int,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes

    xyxy = _xywhn_to_xyxy(boxes, width, height)
    corners = torch.stack(
        [
            xyxy[:, [0, 1]],
            xyxy[:, [2, 1]],
            xyxy[:, [2, 3]],
            xyxy[:, [0, 3]],
        ],
        dim=1,
    )

    angle = torch.tensor(-angle_degrees * np.pi / 180.0, dtype=torch.float32)
    cos_value = torch.cos(angle)
    sin_value = torch.sin(angle)
    center = torch.tensor([width / 2.0, height / 2.0], dtype=torch.float32)

    shifted = corners - center
    rotated_x = shifted[..., 0] * cos_value - shifted[..., 1] * sin_value
    rotated_y = shifted[..., 0] * sin_value + shifted[..., 1] * cos_value
    rotated = torch.stack([rotated_x, rotated_y], dim=-1) + center

    min_xy = rotated.amin(dim=1)
    max_xy = rotated.amax(dim=1)
    clipped = torch.cat([min_xy, max_xy], dim=1)
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clamp(0, width)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clamp(0, height)
    return _xyxy_to_xywhn(clipped, width, height)


def _xywhn_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    converted = boxes.clone()
    converted[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2.0) * width
    converted[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2.0) * height
    converted[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2.0) * width
    converted[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2.0) * height
    return converted


def _xyxy_to_xywhn(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    converted = boxes.clone()
    converted[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2.0) / width
    converted[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2.0) / height
    converted[:, 2] = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) / width
    converted[:, 3] = (boxes[:, 3] - boxes[:, 1]).clamp(min=0) / height
    return converted.clamp(0.0, 1.0)


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
