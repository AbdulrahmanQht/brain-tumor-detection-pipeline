from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from data import DEFAULT_CLASSES, IMAGE_EXTENSIONS, read_yolo_label
from pipeline_logic import ROIExtractor, TumorClassifier, tensor_to_pil


class ResultsManager:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.classes = list(config.get("classes", DEFAULT_CLASSES))
        self.results_dir = Path(config.get("results_dir", "./results/"))
        self.figures_dir = self.results_dir / "figures"
        self.tables_dir = self.results_dir / "tables"
        self.gradcam_dir = self.results_dir / "gradcam"
        self.metrics: dict[str, Any] = {}

        for directory in (self.results_dir, self.figures_dir, self.tables_dir, self.gradcam_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def evaluate_localization(self, yolo_model: Any, test_loader: Iterable) -> dict[str, float]:
        ious: list[float] = []
        detections = 0
        targets_total = 0

        for images, targets in test_loader:
            for image_tensor, target in zip(images, targets):
                gt_boxes = target.get("boxes")
                if gt_boxes is None or gt_boxes.numel() == 0:
                    continue

                targets_total += 1
                image = tensor_to_pil(image_tensor)
                detection = yolo_model.detect(image) if hasattr(yolo_model, "detect") else None
                if detection is None:
                    ious.append(0.0)
                    continue

                detections += 1
                width, height = image.size
                gt_xyxy = _yolo_xywh_to_xyxy(gt_boxes[0], width, height)
                ious.append(compute_iou(detection.bbox, gt_xyxy))

        metrics = {
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "detection_rate": detections / max(targets_total, 1),
            "targets_total": float(targets_total),
            "detections": float(detections),
        }
        self.metrics["localization"] = metrics
        self._save_json(metrics, self.tables_dir / "localization_metrics.json")
        return metrics

    def evaluate_classifiers(
        self,
        predictions: dict[str, Iterable[int | str]] | list[dict[str, Any]],
        ground_truth: Iterable[int | str] | None = None,
    ) -> dict[str, Any]:
        normalized = self._normalize_classifier_inputs(predictions, ground_truth)
        results: dict[str, Any] = {}

        for model_name, payload in normalized.items():
            y_true = [self._label_to_index(label) for label in payload["ground_truth"]]
            y_pred = [self._label_to_index(label) for label in payload["predictions"]]
            labels = list(range(len(self.classes)))

            precision, recall, f1, support = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=labels,
                zero_division=0,
            )
            report = classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=self.classes,
                output_dict=True,
                zero_division=0,
            )
            matrix = confusion_matrix(y_true, y_pred, labels=labels)

            results[model_name] = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
                "per_class": {
                    class_name: {
                        "precision": float(precision[index]),
                        "recall": float(recall[index]),
                        "f1": float(f1[index]),
                        "support": int(support[index]),
                    }
                    for index, class_name in enumerate(self.classes)
                },
                "classification_report": report,
                "confusion_matrix": matrix.tolist(),
            }

        self.metrics["classification"] = results
        self._save_json(results, self.tables_dir / "classification_metrics.json")
        return results

    def plot_confusion_matrices(
        self,
        metrics: dict[str, Any] | None = None,
    ) -> list[Path]:
        metrics = metrics or self.metrics.get("classification")
        if not metrics:
            metrics_path = self.tables_dir / "classification_metrics.json"
            if not metrics_path.exists():
                return []
            metrics = self._load_json(metrics_path)

        output_paths: list[Path] = []
        for model_name, payload in metrics.items():
            matrix = np.asarray(payload["confusion_matrix"])
            plt.figure(figsize=(7, 6))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.classes,
                yticklabels=self.classes,
                cbar=False,
            )
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{model_name} Confusion Matrix")
            plt.tight_layout()

            output_path = self.figures_dir / f"{model_name}_confusion_matrix.png"
            plt.savefig(output_path, dpi=200)
            plt.close()
            output_paths.append(output_path)

        return output_paths

    def plot_training_curves(self, history_dict: dict[str, dict[str, list[float]]]) -> list[Path]:
        output_paths: list[Path] = []
        for model_name, history in history_dict.items():
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(history.get("train_loss", []), label="Train")
            axes[0].plot(history.get("val_loss", []), label="Validation")
            axes[0].set_title(f"{model_name} Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()

            axes[1].plot(history.get("train_acc", []), label="Train")
            axes[1].plot(history.get("val_acc", []), label="Validation")
            axes[1].set_title(f"{model_name} Accuracy")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].legend()

            fig.tight_layout()
            output_path = self.figures_dir / f"{model_name}_training_curves.png"
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
            output_paths.append(output_path)

        return output_paths

    def plot_before_after_samples(self, n: int = 5) -> list[Path]:
        dataset_path = Path(self.config.get("dataset_path", "./dataset/"))
        extractor = ROIExtractor(self.config)
        output_paths: list[Path] = []
        sample_paths = self._sample_image_paths(dataset_path / "test" / "images", n)

        for index, image_path in enumerate(sample_paths, start=1):
            image = Image.open(image_path).convert("RGB")
            label_path = dataset_path / "test" / "labels" / f"{image_path.stem}.txt"
            boxes, labels = read_yolo_label(label_path)
            class_name, crop = extractor._crop_from_labels(image, boxes, labels)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(image)
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(crop)
            axes[1].set_title(f"ROI: {class_name}")
            axes[1].axis("off")
            fig.tight_layout()

            output_path = self.figures_dir / f"before_after_{index:02d}.png"
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
            output_paths.append(output_path)

        return output_paths

    def generate_comparative_table(
        self,
        metrics: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        metrics = metrics or self.metrics.get("classification")
        if not metrics:
            metrics_path = self.tables_dir / "classification_metrics.json"
            if metrics_path.exists():
                metrics = self._load_json(metrics_path)
            else:
                metrics = {}

        rows: list[dict[str, Any]] = []
        for model_name, payload in metrics.items():
            row = {
                "model": model_name,
                "accuracy": payload.get("accuracy", 0.0),
                "macro_f1": payload.get("macro_f1", 0.0),
                "weighted_f1": payload.get("weighted_f1", 0.0),
            }
            for class_name in self.classes:
                row[f"{class_name}_f1"] = payload.get("per_class", {}).get(class_name, {}).get("f1", 0.0)
            rows.append(row)

        table = pd.DataFrame(rows)
        output_path = self.tables_dir / "comparative_table.csv"
        table.to_csv(output_path, index=False)
        return table

    def generate_gradcam(
        self,
        classifier: TumorClassifier,
        test_images: Iterable[Image.Image | torch.Tensor | str | Path],
        n: int = 10,
    ) -> list[Path]:
        output_paths: list[Path] = []
        target_layer = self._get_gradcam_target_layer(classifier)
        model = classifier.model.to(classifier.device)
        model.eval()

        for index, image_input in enumerate(test_images):
            if index >= n:
                break

            pil_image = _as_pil_image(image_input)
            input_tensor = _preprocess_for_classifier(pil_image, self.config).unsqueeze(0).to(classifier.device)
            heatmap, predicted_index = _compute_gradcam(model, target_layer, input_tensor)
            overlay = _overlay_heatmap(pil_image.resize((self.config.get("roi_size", 224), self.config.get("roi_size", 224))), heatmap)

            class_name = classifier.classes[predicted_index]
            output_path = self.gradcam_dir / f"{classifier.model_name}_{index + 1:02d}_{class_name}.png"
            Image.fromarray(overlay).save(output_path)
            output_paths.append(output_path)

        return output_paths

    def generate_benchmark_table(self, literature_results: Iterable[dict[str, Any]]) -> pd.DataFrame:
        rows = list(literature_results)
        best_model = None
        if self.metrics.get("classification"):
            best_model = max(
                self.metrics["classification"].items(),
                key=lambda item: item[1].get("macro_f1", 0.0),
            )

        if best_model is not None:
            model_name, metrics = best_model
            rows.append(
                {
                    "study": "This work",
                    "model": model_name,
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "weighted_f1": metrics.get("weighted_f1"),
                }
            )

        table = pd.DataFrame(rows)
        output_path = self.tables_dir / "benchmark_table.csv"
        table.to_csv(output_path, index=False)
        return table

    def save_all(self, output_dir: str | Path | None = None) -> None:
        if output_dir is not None:
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            self._save_json(self.metrics, output / "all_metrics.json")
        else:
            self._save_json(self.metrics, self.results_dir / "all_metrics.json")

    def _normalize_classifier_inputs(
        self,
        predictions: dict[str, Iterable[int | str]] | list[dict[str, Any]],
        ground_truth: Iterable[int | str] | None,
    ) -> dict[str, dict[str, list[int | str]]]:
        if isinstance(predictions, list):
            by_model: dict[str, dict[str, list[int | str]]] = {}
            for item in predictions:
                target = item.get("target")
                classifications = item.get("classification", {})
                for model_name, predicted_label in classifications.items():
                    by_model.setdefault(model_name, {"ground_truth": [], "predictions": []})
                    by_model[model_name]["ground_truth"].append(target)
                    by_model[model_name]["predictions"].append(predicted_label)
            return by_model

        if ground_truth is None:
            raise ValueError("ground_truth is required when predictions are provided as a dictionary.")

        truth = list(ground_truth)
        return {
            model_name: {"ground_truth": truth, "predictions": list(model_predictions)}
            for model_name, model_predictions in predictions.items()
        }

    def _label_to_index(self, label: int | str | None) -> int:
        if label is None:
            return self.classes.index("no_tumor")
        if isinstance(label, str):
            normalized = label.strip().lower().replace(" ", "_")
            return self.classes.index(normalized)
        return int(label)

    def _sample_image_paths(self, images_dir: Path, n: int) -> list[Path]:
        paths = [
            path
            for path in sorted(images_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return paths[:n]

    def _get_gradcam_target_layer(self, classifier: TumorClassifier) -> torch.nn.Module:
        model_name = classifier.model_name.lower()
        model = classifier.model
        if model_name == "resnet50":
            return model.layer4[-1]
        if model_name in {"mobilenet_v2", "efficientnet_b0"}:
            return model.features[-1]
        raise ValueError(f"Unsupported classifier for Grad-CAM: {classifier.model_name}")

    def _save_json(self, payload: Any, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2)

    def _load_json(self, path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def compute_iou(
    box_a: Iterable[float],
    box_b: Iterable[float],
) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _compute_gradcam(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> tuple[np.ndarray, int]:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        logits = model(input_tensor)
        predicted_index = int(torch.argmax(logits, dim=1).item())
        score = logits[:, predicted_index].sum()
        model.zero_grad(set_to_none=True)
        score.backward()

        activation = activations[-1].detach()
        gradient = gradients[-1].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(1), size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, predicted_index
    finally:
        forward_handle.remove()
        backward_handle.remove()


def _overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_array, 1.0 - alpha, colored, alpha, 0)


def _as_pil_image(image_input: Image.Image | torch.Tensor | str | Path) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, torch.Tensor):
        return tensor_to_pil(image_input).convert("RGB")
    return Image.open(image_input).convert("RGB")


def _preprocess_for_classifier(image: Image.Image, config: dict[str, Any]) -> torch.Tensor:
    roi_size = int(config.get("roi_size", 224))
    image = image.convert("RGB").resize((roi_size, roi_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def _yolo_xywh_to_xyxy(box: torch.Tensor, width: int, height: int) -> tuple[float, float, float, float]:
    x_center, y_center, box_width, box_height = [float(value) for value in box.tolist()]
    return (
        (x_center - box_width / 2.0) * width,
        (y_center - box_height / 2.0) * height,
        (x_center + box_width / 2.0) * width,
        (y_center + box_height / 2.0) * height,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
