from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score

from data import compute_class_weights, load_roi_dataset, load_yolo_dataset
from pipeline_logic import BrainTumorPipeline, ROIExtractor, TumorClassifier, TumorLocator


CONFIG: dict[str, Any] = {
    # Data
    "image_size": 640,
    "roi_size": 224,
    "classes": ["glioma", "meningioma", "pituitary", "no_tumor"],
    "num_classes": 4,
    "num_workers": 2,
    "pin_memory": torch.cuda.is_available(),
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Paths
    "dataset_path": "./dataset/",
    "roi_dataset_path": "./roi_dataset/",
    "weights_dir": "./weights/",
    "results_dir": "./results/",

    # YOLO
    "yolo_model": "yolo11n.pt",
    "yolo_weights": None,
    "yolo_epochs": 100,
    "yolo_patience": 20,
    "yolo_conf_thresh": 0.45,
    "yolo_iou_thresh": 0.50,
    "yolo_metric": "mAP_0.5:0.95",
    "use_cbam": True,
    "cbam_min_layer_index": 10,

    # ROI cropping
    "roi_padding": 0.10,
    "force_rebuild_roi": False,

    # Classifiers
    "classifiers": ["resnet50", "mobilenet_v2", "efficientnet_b0"],
    "batch_size": 32,
    "learning_rate": 1e-4,
    "learning_rate_ft": 1e-5,
    "head_epochs": 10,
    "finetune_epochs": 20,
    "dropout_rate": 0.4,
    "use_pretrained": True,
    "classifier_weights": {},

    # Imbalance
    "use_class_weights": True,

    # Stage flags
    "load_yolo_data": True,
    "train_yolo": True,
    "build_roi_dataset": True,
    "train_classifiers": True,
    "run_pipeline_on_test": True,
    "run_results_manager": True,

    # Optional optimization / reports
    "no_tumor_fallback": True,
    "run_optuna": True,
    "optuna_trials": 15,
    "run_gradcam": True,
}


def main(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = deepcopy(CONFIG)
    if config:
        cfg.update(config)

    set_seed(int(cfg.get("seed", 42)))
    ensure_directories(cfg)
    validate_dataset_paths(cfg)

    artifacts: dict[str, Any] = {"config": cfg, "histories": {}, "classifier_metrics": {}}

    if cfg.get("load_yolo_data", True):
        artifacts["yolo_loaders"] = load_yolo_dataset(cfg)

    # 1. Train YOLO first
    locator = None
    if cfg.get("train_yolo", False):
        locator = TumorLocator(cfg)
        artifacts["yolo_results"] = locator.train()
        yolo_path = Path(cfg["weights_dir"]) / "tumor_locator.pt"
        locator.save_weights(yolo_path)
        cfg["yolo_weights"] = str(yolo_path)

    # 2. Build ROI dataset from final YOLO weights
    if cfg.get("build_roi_dataset", False):
        extractor = ROIExtractor(cfg)
        artifacts["roi_counts"] = extractor.build_roi_dataset(
            cfg["dataset_path"],
            cfg["roi_dataset_path"],
            force_rebuild=cfg.get("force_rebuild_roi", False),
        )
        save_json(artifacts["roi_counts"], Path(cfg["results_dir"]) / "roi_counts.json")

    # 3. Run Optuna AFTER ROI dataset exists
    if cfg.get("run_optuna", False):
        artifacts["optuna"] = run_optuna_search(cfg)

    # 4. Train all classifiers with their own best params
    roi_loaders = None
    if cfg.get("train_classifiers", False):
        roi_loaders = load_roi_dataset(cfg)
        train_loader, val_loader, _ = roi_loaders
        class_weights = build_class_weights(cfg)
        optuna_results = artifacts.get("optuna", {})

        for model_name in cfg["classifiers"]:
            classifier_cfg = deepcopy(cfg)

            # Apply this model's own best Optuna params if available
            if model_name in optuna_results:
                best = optuna_results[model_name]["best_params"]
                classifier_cfg.update(best)
                print(f"\n  Using Optuna params for {model_name}: {best}")

            # Reload DataLoader in case batch_size changed for this model
            model_train_loader, model_val_loader, _ = load_roi_dataset(classifier_cfg)

            classifier = TumorClassifier(model_name, classifier_cfg)
            history = classifier.train(model_train_loader, model_val_loader, class_weights)
            metrics = evaluate_classifier(classifier, model_val_loader)
            weights_path = Path(cfg["weights_dir"]) / f"{model_name}.pth"
            classifier.save_weights(weights_path)

            artifacts["histories"][model_name] = history
            artifacts["classifier_metrics"][model_name] = metrics
            cfg.setdefault("classifier_weights", {})[model_name] = str(weights_path)

        save_json(artifacts["histories"], Path(cfg["results_dir"]) / "training_histories.json")
        save_json(artifacts["classifier_metrics"], Path(cfg["results_dir"]) / "classifier_metrics.json")

    # 5. Run pipeline on test set
    if cfg.get("run_pipeline_on_test", False):
        if roi_loaders is None:
            roi_loaders = load_roi_dataset(cfg)
        _, _, test_loader = roi_loaders
        pipeline = BrainTumorPipeline(cfg)
        artifacts["pipeline_predictions"] = pipeline.run_batch(test_loader)
        save_json(
            _json_safe(artifacts["pipeline_predictions"]),
            Path(cfg["results_dir"]) / "pipeline_predictions.json",
        )

    # 6. Generate all results
    if cfg.get("run_results_manager", False):
        artifacts["results_manager"] = run_results_manager(cfg, artifacts, locator)

    save_json(_json_safe(cfg), Path(cfg["results_dir"]) / "config_used.json")
    return artifacts

def run_optuna_search(config: dict[str, Any]) -> dict[str, Any]:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    all_results: dict[str, Any] = {}

    for model_name in config.get("classifiers", ["resnet50", "mobilenet_v2", "efficientnet_b0"]):
        print(f"\n{'='*60}")
        print(f"  Optuna search: {model_name.upper()} ({config.get('optuna_trials', 15)} trials)")
        print(f"{'='*60}")

        def objective(trial: optuna.Trial) -> float:
            trial_config = deepcopy(config)
            trial_config.update({
                "learning_rate":   trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "dropout_rate":    trial.suggest_float("dropout_rate", 0.2, 0.6),
                "batch_size":      trial.suggest_categorical("batch_size", [16, 32, 64]),
                "classifiers":     [model_name],
                "head_epochs":     5,
                "finetune_epochs": 5,
            })
            train_loader, val_loader, _ = load_roi_dataset(trial_config)
            class_weights = build_class_weights(trial_config)
            classifier = TumorClassifier(model_name, trial_config)
            classifier.train(train_loader, val_loader, class_weights)
            return evaluate_classifier(classifier, val_loader)["macro_f1"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=int(config.get("optuna_trials", 15)))

        all_results[model_name] = {
            "best_params": study.best_params,
            "best_value":  study.best_value,
        }
        print(f"  Best macro F1: {study.best_value:.4f}")
        print(f"  Best params:   {study.best_params}")

    save_json(all_results, Path(config["results_dir"]) / "optuna_best.json")
    return all_results


def evaluate_classifier(classifier: TumorClassifier, data_loader) -> dict[str, float]:
    classifier.model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(classifier.device)
            logits = classifier.model(images)
            preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
            y_pred.extend(int(pred) for pred in preds)
            y_true.extend(int(label) for label in labels.tolist())

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if y_true else 0.0,
    }


def build_class_weights(config: dict[str, Any]) -> torch.Tensor | None:
    if not config.get("use_class_weights", True):
        return None
    train_dir = Path(config["roi_dataset_path"]) / "train"
    return compute_class_weights(train_dir, classes=config["classes"], device=config["device"])


def run_results_manager(
    config: dict[str, Any],
    artifacts: dict[str, Any],
    locator: TumorLocator | None,
) -> Any:
    try:
        from results_manager import ResultsManager
    except ImportError as exc:
        raise RuntimeError("results_manager.py is not ready yet.") from exc

    manager = ResultsManager(config)
    if hasattr(manager, "plot_training_curves") and artifacts.get("histories"):
        manager.plot_training_curves(artifacts["histories"])
    if hasattr(manager, "save_all"):
        manager.save_all(config["results_dir"])
    return manager


def ensure_directories(config: dict[str, Any]) -> None:
    for key in ("weights_dir", "results_dir"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)


def validate_dataset_paths(config: dict[str, Any]) -> None:
    dataset_root = Path(config["dataset_path"])
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing YOLO data file: {data_yaml}")
    for split in ("train", "valid", "test"):
        for child in ("images", "labels"):
            path = dataset_root / split / child
            if not path.exists():
                raise FileNotFoundError(f"Missing dataset directory: {path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2)


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


if __name__ == "__main__":
    print("cuda" if torch.cuda.is_available() else "cpu")
    main()
