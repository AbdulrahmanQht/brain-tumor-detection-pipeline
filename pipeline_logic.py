from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from data import (
    DEFAULT_CLASSES,
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SPLITS,
    YOLO_ID_TO_CLASS,
    normalize_split,
    read_yolo_label,
)


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int | None = None


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        max_out = self.mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x) * x
        return self.spatial_attention(x) * x


class TumorLocator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model_name = config.get("yolo_model", "yolo11n.pt")
        self.conf_thresh = float(config.get("yolo_conf_thresh", 0.45))
        self.iou_thresh = float(config.get("yolo_iou_thresh", 0.50))
        self.image_size = int(config.get("image_size", 640))
        self.model = self._load_yolo_model(str(config.get("yolo_weights") or self.model_name))
        if bool(config.get("use_cbam", True)):
            self._insert_cbam_after_c2f()

    def _load_yolo_model(self, model_name: str):
        from ultralytics import YOLO

        return YOLO(model_name)

    def _insert_cbam_after_c2f(self) -> None:
        """
        Best-effort CBAM insertion after YOLO C2f modules.

        Ultralytics exposes the PyTorch graph as model.model.model. The exact
        YOLOv11 module layout can change by package version, so this keeps the
        modification conservative: only sequential top-level modules with a
        discoverable output channel count are wrapped.
        """

        try:
            layers = self.model.model.model
        except AttributeError:
            return

        min_layer_index = int(self.config.get("cbam_min_layer_index", 10))
        for index, layer in enumerate(list(layers)):
            if index < min_layer_index:
                continue
            if "C2f" not in layer.__class__.__name__:
                continue
            channels = _infer_out_channels(layer)
            if channels is None:
                continue
            layers[index] = nn.Sequential(layer, CBAM(channels))

    def train(self, dataloader: Any | None = None) -> Any:
        data_yaml = Path(self.config.get("dataset_path", "./dataset/")) / "data.yaml"
        if isinstance(dataloader, (str, Path)):
            data_yaml = Path(dataloader)

        results = self.model.train(
            data=str(data_yaml),
            imgsz=self.image_size,
            epochs=int(self.config.get("yolo_epochs", 50)),
            patience=int(self.config.get("yolo_patience", 10)),
            batch=int(self.config.get("batch_size", 32)),
            workers=int(self.config.get("num_workers", 2)),
            project=str(Path(self.config.get("results_dir", "./results/")) / "yolo"),
            name="tumor_locator",
            exist_ok=True,
        )
        return results

    @torch.inference_mode()
    def detect(self, image: Image.Image | np.ndarray | str | Path) -> Detection | None:
        results = self.model.predict(
            source=image,
            imgsz=self.image_size,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
        )
        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        confidences = boxes.conf.detach().cpu()
        best_index = int(torch.argmax(confidences).item())
        confidence = float(confidences[best_index].item())
        if confidence < self.conf_thresh:
            return None

        xyxy = boxes.xyxy[best_index].detach().cpu().tolist()
        class_id = None
        if getattr(boxes, "cls", None) is not None:
            class_id = int(boxes.cls[best_index].detach().cpu().item())
        return Detection(
            bbox=tuple(float(value) for value in xyxy),
            confidence=confidence,
            class_id=class_id,
        )

    def save_weights(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(destination))

    def load_weights(self, path: str | Path) -> None:
        self.model = self._load_yolo_model(str(path))


class ROIExtractor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.roi_size = int(config.get("roi_size", 224))
        self.padding = float(config.get("roi_padding", 0.10))
        self.classes = list(config.get("classes", DEFAULT_CLASSES))

    def get_padded_crop(
        self,
        image: Image.Image,
        bbox: Iterable[float],
        padding: float | None = None,
    ) -> Image.Image:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        image_width, image_height = image.size
        pad = self.padding if padding is None else padding

        box_width = x2 - x1
        box_height = y2 - y1
        x1_pad = max(0.0, x1 - box_width * pad)
        y1_pad = max(0.0, y1 - box_height * pad)
        x2_pad = min(float(image_width), x2 + box_width * pad)
        y2_pad = min(float(image_height), y2 + box_height * pad)

        crop = image.crop((x1_pad, y1_pad, x2_pad, y2_pad))
        return crop.resize((self.roi_size, self.roi_size), Image.BILINEAR)

    def build_roi_dataset(
        self,
        dataset_path: str | Path,
        roi_path: str | Path,
        force_rebuild: bool | None = None,
    ) -> dict[str, dict[str, int]]:
        dataset_root = Path(dataset_path)
        roi_root = Path(roi_path)
        force = bool(self.config.get("force_rebuild_roi", False) if force_rebuild is None else force_rebuild)

        if roi_root.exists() and force:
            shutil.rmtree(roi_root)
        roi_root.mkdir(parents=True, exist_ok=True)

        counts = {split: {class_name: 0 for class_name in self.classes} for split in SPLITS}
        for split in SPLITS:
            split = normalize_split(split)
            for class_name in self.classes:
                (roi_root / split / class_name).mkdir(parents=True, exist_ok=True)

            images_dir = dataset_root / split / "images"
            labels_dir = dataset_root / split / "labels"
            if not images_dir.exists() or not labels_dir.exists():
                raise FileNotFoundError(f"Missing YOLO split directories under {dataset_root / split}")

            for image_path in sorted(images_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                image = Image.open(image_path).convert("RGB")
                label_path = labels_dir / f"{image_path.stem}.txt"
                boxes, labels = read_yolo_label(label_path)

                class_name, crop = self._crop_from_labels(image, boxes, labels)
                destination = roi_root / split / class_name / f"{image_path.stem}.jpg"
                crop.save(destination, quality=95)
                counts[split][class_name] += 1

        return counts

    def _crop_from_labels(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[str, Image.Image]:
        if boxes.numel() == 0 or labels.numel() == 0:
            return "no_tumor", self._center_crop(image)

        best_index = _largest_yolo_box_index(boxes)
        class_id = int(labels[best_index].item())
        class_name = YOLO_ID_TO_CLASS.get(class_id, "no_tumor")
        if class_name == "no_tumor":
            return class_name, self._center_crop(image)

        bbox = _yolo_xywh_to_xyxy(boxes[best_index], *image.size)
        return class_name, self.get_padded_crop(image, bbox)

    def _center_crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = min(width, height)
        left = (width - side) / 2.0
        top = (height - side) / 2.0
        crop = image.crop((left, top, left + side, top + side))
        return crop.resize((self.roi_size, self.roi_size), Image.BILINEAR)


class TumorClassifier:
    def __init__(self, model_name: str, config: dict[str, Any]) -> None:
        self.model_name = model_name
        self.config = config
        self.classes = list(config.get("classes", DEFAULT_CLASSES))
        self.num_classes = int(config.get("num_classes", len(self.classes)))
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.build_model().to(self.device)

    def build_model(self) -> nn.Module:
        from torchvision import models

        dropout_rate = float(self.config.get("dropout_rate", 0.4))
        model_name = self.model_name.lower()
        use_pretrained = bool(self.config.get("use_pretrained", True))

        if model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = _classification_head(in_features, self.num_classes, dropout_rate)
            return model

        if model_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if use_pretrained else None
            model = models.mobilenet_v2(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier = _classification_head(in_features, self.num_classes, dropout_rate)
            return model

        if model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier = _classification_head(in_features, self.num_classes, dropout_rate)
            return model

        raise ValueError(f"Unsupported classifier: {self.model_name}")
    
    def _set_frozen_bn_eval(self) -> None:
        """Keep frozen BatchNorm layers in eval mode to preserve pretrained running stats."""
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # Only force eval if its parameters are frozen
                if not any(p.requires_grad for p in module.parameters()):
                    module.eval()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor | None = None,
    ) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device) if class_weights is not None else None)
        
        print(f"\n{'='*60}")
        print(f"  Training: {self.model_name}")
        print(f"{'='*60}")


        self._freeze_backbone()
        optimizer = torch.optim.Adam(
            (param for param in self.model.parameters() if param.requires_grad),
            lr=float(self.config.get("learning_rate", 1e-4)),
        )
        
        print(f"  Stage 1 — Head training ({self.config.get('head_epochs', 10)} epochs, lr={self.config.get('learning_rate', 1e-4)})")

        self._run_epochs(
            train_loader,
            val_loader,
            criterion,
            optimizer,
            int(self.config.get("head_epochs", 10)),
            history,
        )

        self._unfreeze_top_blocks()
        optimizer = torch.optim.Adam(
            (param for param in self.model.parameters() if param.requires_grad),
            lr=float(self.config.get("learning_rate_ft", 1e-5)),
        )
        
        print(f"  Stage 2 — Fine-tuning ({self.config.get('finetune_epochs', 20)} epochs, lr={self.config.get('learning_rate_ft', 1e-5)})")
        
        self._run_epochs(
            train_loader,
            val_loader,
            criterion,
            optimizer,
            int(self.config.get("finetune_epochs", 20)),
            history,
        )
        return history

    @torch.inference_mode()
    def predict(self, roi_image: Image.Image | torch.Tensor) -> str:
        self.model.eval()
        if isinstance(roi_image, Image.Image):
            image_tensor = preprocess_roi_image(roi_image, int(self.config.get("roi_size", 224)))
        else:
            image_tensor = roi_image
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        logits = self.model(image_tensor.to(self.device))
        class_index = int(torch.argmax(logits, dim=1).item())
        return self.classes[class_index]

    def save_weights(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": self.model_name,
                "classes": self.classes,
                "state_dict": self.model.state_dict(),
            },
            destination,
        )

    def load_weights(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict)

    def _run_epochs(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        history: dict[str, list[float]],
    ) -> None:
        total_epochs_so_far = len(history["train_loss"])
        for epoch in range(epochs):
            t0 = time.time()
            train_loss, train_acc = self._run_one_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self._evaluate_loader(val_loader, criterion)
            elapsed = time.time() - t0
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            print(
            f"  [{self.model_name}] "
            f"Epoch {total_epochs_so_far + epoch + 1:>3} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss:   {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

    def _run_one_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float]:
        self.model.train()
        self._set_frozen_bn_eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
            total += batch_size

        return total_loss / max(total, 1), correct / max(total, 1)

    @torch.inference_mode()
    def _evaluate_loader(self, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
            total += batch_size

        return total_loss / max(total, 1), correct / max(total, 1)

    def _freeze_backbone(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        for param in _classifier_parameters(self.model, self.model_name):
            param.requires_grad = True

    def _unfreeze_top_blocks(self) -> None:
        self._freeze_backbone()
        model_name = self.model_name.lower()
        modules: list[nn.Module] = []

        if model_name == "resnet50":
            modules = list(self.model.layer4.children())[-2:]
        elif model_name == "mobilenet_v2":
            modules = list(self.model.features.children())[-2:]
        elif model_name == "efficientnet_b0":
            modules = list(self.model.features.children())[-2:]

        for module in modules:
            for param in module.parameters():
                param.requires_grad = True


class BrainTumorPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.classes = list(config.get("classes", DEFAULT_CLASSES))
        self.locator = TumorLocator(config)
        self.extractor = ROIExtractor(config)
        self.classifiers = {
            model_name: TumorClassifier(model_name, config)
            for model_name in config.get("classifiers", ["resnet50", "mobilenet_v2", "efficientnet_b0"])
        }
        classifier_weights = config.get("classifier_weights", {})
        for model_name, weights_path in classifier_weights.items():
            if model_name in self.classifiers and weights_path:
                self.classifiers[model_name].load_weights(weights_path)

    def run(self, image: Image.Image | str | Path) -> dict[str, Any]:
        pil_image = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
        detection = self.locator.detect(pil_image)
        if detection is None and bool(self.config.get("no_tumor_fallback", True)):
            return {
                "yolo_bbox": None,
                "yolo_confidence": None,
                "classification": {name: "no_tumor" for name in self.classifiers},
            }

        if detection is None:
            crop = self.extractor._center_crop(pil_image)
            bbox = None
            confidence = None
        else:
            crop = self.extractor.get_padded_crop(pil_image, detection.bbox)
            bbox = detection.bbox
            confidence = detection.confidence

        return {
            "yolo_bbox": bbox,
            "yolo_confidence": confidence,
            "classification": {
                name: classifier.predict(crop)
                for name, classifier in self.classifiers.items()
            },
        }

    def run_batch(self, dataloader: DataLoader) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
            else:
                images, labels = batch, None

            for index, image_tensor in enumerate(images):
                image = tensor_to_pil(image_tensor)
                result = self.run(image)
                if labels is not None:
                    result["target"] = _batch_target_at(labels, index)
                outputs.append(result)
        return outputs


def preprocess_roi_image(image: Image.Image, roi_size: int = 224) -> torch.Tensor:
    image = image.convert("RGB").resize((roi_size, roi_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def _classification_head(in_features: int, num_classes: int, dropout_rate: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(512, num_classes),
    )


def _classifier_parameters(model: nn.Module, model_name: str):
    model_name = model_name.lower()
    if model_name == "resnet50":
        return model.fc.parameters()
    if model_name in {"mobilenet_v2", "efficientnet_b0"}:
        return model.classifier.parameters()
    raise ValueError(f"Unsupported classifier: {model_name}")


def _infer_out_channels(layer: nn.Module) -> int | None:
    for module in reversed(list(layer.modules())):
        channels = getattr(module, "out_channels", None)
        if isinstance(channels, int):
            return channels
    return None


def _largest_yolo_box_index(boxes: torch.Tensor) -> int:
    areas = boxes[:, 2] * boxes[:, 3]
    return int(torch.argmax(areas).item())


def _yolo_xywh_to_xyxy(box: torch.Tensor, width: int, height: int) -> tuple[float, float, float, float]:
    x_center, y_center, box_width, box_height = [float(value) for value in box.tolist()]
    x1 = (x_center - box_width / 2.0) * width
    y1 = (y_center - box_height / 2.0) * height
    x2 = (x_center + box_width / 2.0) * width
    y2 = (y_center + box_height / 2.0) * height
    return x1, y1, x2, y2


def _batch_target_at(targets: Any, index: int) -> Any:
    if isinstance(targets, torch.Tensor):
        value = targets[index]
        return int(value.item()) if value.numel() == 1 else value.detach().cpu()
    if isinstance(targets, list):
        return targets[index]
    if isinstance(targets, tuple):
        return targets[index]
    return targets
