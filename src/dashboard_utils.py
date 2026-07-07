from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    outputs_dir: Path
    results_dir: Path
    figures_dir: Path
    models_dir: Path
    train_crops_dir: Path
    validation_crops_dir: Path


def get_project_paths(root: Path | str) -> ProjectPaths:
    root_path = Path(root)
    outputs_dir = root_path / "outputs"
    return ProjectPaths(
        root=root_path,
        outputs_dir=outputs_dir,
        results_dir=outputs_dir / "results",
        figures_dir=outputs_dir / "figures",
        models_dir=outputs_dir / "models",
        train_crops_dir=root_path / "crops" / "train",
        validation_crops_dir=root_path / "crops" / "validation",
    )


def safe_read_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path, **kwargs)


def report_accuracy(report_df: pd.DataFrame) -> float | None:
    if report_df.empty:
        return None
    if "accuracy" in report_df.index and "precision" in report_df.columns:
        return float(report_df.loc["accuracy", "precision"])
    if "accuracy" in report_df.columns:
        return float(report_df["accuracy"].iloc[0])
    return None


def model_comparison_records(comparison_df: pd.DataFrame) -> list[dict[str, float | str]]:
    if comparison_df.empty:
        return []

    records: list[dict[str, float | str]] = []
    for _, row in comparison_df.iterrows():
        model = row.get("model", row.get("Model", "Model"))
        accuracy = row.get("accuracy", row.get("Accuracy", None))
        macro_f1 = row.get("macro_f1", row.get("Macro F1", row.get("macro avg", None)))
        record: dict[str, float | str] = {"Model": str(model)}
        if accuracy is not None:
            record["Accuracy"] = float(accuracy)
        if macro_f1 is not None:
            record["Macro F1"] = float(macro_f1)
        records.append(record)
    return records


def list_class_names(crops_dir: Path | str) -> list[str]:
    root = Path(crops_dir)
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def sample_images_for_class(crops_dir: Path | str, class_name: str, limit: int = 12) -> list[Path]:
    class_dir = Path(crops_dir) / class_name
    if not class_dir.exists():
        return []
    images = [
        path
        for path in sorted(class_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return images[:limit]


def top_probabilities(
    class_names: Iterable[str],
    probabilities: Iterable[float],
    limit: int = 6,
) -> list[tuple[str, float]]:
    pairs = [(name, float(prob)) for name, prob in zip(class_names, probabilities)]
    return sorted(pairs, key=lambda item: item[1], reverse=True)[:limit]


def resize_cam(cam, width: int, height: int) -> np.ndarray:
    import cv2

    cam_array = np.asarray(cam, dtype=np.float32)
    resized = cv2.resize(cam_array, (width, height), interpolation=cv2.INTER_LINEAR)
    resized -= resized.min()
    max_value = resized.max()
    if max_value > 0:
        resized /= max_value
    return resized


def load_resnet18_checkpoint(model_path: Path | str, device: str = "cpu"):
    import torch
    import torch.nn as nn
    from torchvision import models

    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    img_size = int(checkpoint.get("img_size", 224))

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, img_size


def predict_image(model, image: Image.Image, class_names: list[str], img_size: int, device: str = "cpu"):
    import torch
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    top_class, top_score = top_probabilities(class_names, probabilities, limit=1)[0]
    return top_class, top_score, probabilities


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class: int):
        import torch

        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[:, target_class].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max().clamp(min=1e-8)
        return cam.detach().cpu().numpy()

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def gradcam_overlay(model, image: Image.Image, target_class: int, img_size: int, device: str = "cpu") -> Image.Image:
    import cv2
    import torch
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    gradcam = GradCAM(model, model.layer4[-1])
    try:
        cam = gradcam(input_tensor, target_class)
    finally:
        gradcam.close()

    original = image.convert("RGB").resize((img_size, img_size))
    original_np = np.array(original)
    cam = resize_cam(cam, width=img_size, height=img_size)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.55 * original_np + 0.45 * heatmap)
    return Image.fromarray(overlay)
