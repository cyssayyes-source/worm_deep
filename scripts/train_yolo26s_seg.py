"""Train the worm segmentation model with Ultralytics YOLO26s-seg."""

import argparse
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


MIN_ULTRALYTICS = (8, 4, 52)
MODEL_ALIASES = {
    "yolov26s-seg": "yolo26s-seg.pt",
    "yolov26s-seg.pt": "yolo26s-seg.pt",
    "yolo26s-seg": "yolo26s-seg.pt",
}


def version_tuple(version: str) -> tuple[int, int, int]:
    parts = []
    for item in version.split(".")[:3]:
        digits = "".join(ch for ch in item if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def resolve_weights(weights: str) -> str:
    weights = MODEL_ALIASES.get(weights.lower(), weights)
    local = Path(weights)
    if local.exists():
        return str(local)
    pretrained = Path("models") / "pretrained" / weights
    if pretrained.exists():
        return str(pretrained)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Train worm segmentation with YOLO26s-seg.")
    parser.add_argument(
        "--weights",
        default="models/pretrained/yolo26s-seg.pt",
        help="YOLO26s-seg pretrained weights or model name. 'yolov26s-seg' is accepted as an alias.",
    )
    parser.add_argument(
        "--data",
        default="label/combined_yolo_dataset_reviewed/dataset.yaml",
        help="YOLO segmentation dataset.yaml",
    )
    parser.add_argument("--project", default="worm_yolo_results", help="Training output root")
    parser.add_argument("--name", default="train_yolo26s_seg", help="Training run name")
    parser.add_argument("--epochs", type=int, default=300, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=80, help="Early stopping patience")
    parser.add_argument("--batch", type=int, default=1, help="Batch size. Use 1 on 8GB GPUs if unsure.")
    parser.add_argument("--imgsz", type=int, default=1376, help="Image size")
    parser.add_argument("--device", default=0, help="GPU id or 'cpu'")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers. Keep 0 on Windows.")
    parser.add_argument("--optimizer", default="auto", help="Optimizer. Use AdamW/SGD to force --lr0.")
    parser.add_argument("--lr0", type=float, default=None, help="Initial LR. Only passed when optimizer is not auto.")
    parser.add_argument("--degrees", type=float, default=180.0, help="Random rotation range")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability")
    parser.add_argument("--flipud", type=float, default=0.5, help="Vertical flip probability")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--single-cls", action="store_true", help="Train as one class")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting/reusing an existing run directory")
    args = parser.parse_args()

    if version_tuple(ultralytics.__version__) < MIN_ULTRALYTICS:
        raise RuntimeError(
            f"YOLO26 requires ultralytics>={'.'.join(map(str, MIN_ULTRALYTICS))}; "
            f"current version is {ultralytics.__version__}. "
            "Run: python -m pip install -U \"ultralytics>=8.4.52\""
        )

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "degrees": args.degrees,
        "fliplr": args.fliplr,
        "flipud": args.flipud,
        "seed": args.seed,
        "save": True,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "pretrained": True,
        "augment": True,
        "single_cls": args.single_cls,
    }
    if args.optimizer.lower() != "auto" and args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    elif args.optimizer.lower() == "auto" and args.lr0 is not None:
        print("Note: --optimizer auto ignores --lr0; choose --optimizer AdamW/SGD to force it.")

    model = YOLO(resolve_weights(args.weights))
    model.train(**train_kwargs)
    print("CUDA available:", torch.cuda.is_available())
    print("Ultralytics version:", ultralytics.__version__)


if __name__ == "__main__":
    main()
