import argparse

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train a YOLOv8 segmentation model.")
    parser.add_argument(
        "--weights",
        default="models/worm_yolo_best.pt",
        help="YOLO weights to start from",
    )
    parser.add_argument("--data", default="label/combined_yolo_dataset/dataset.yaml", help="YOLO dataset config")
    parser.add_argument("--project", default="worm_yolo_results", help="Project output folder")
    parser.add_argument("--name", default="train_combined_finetune", help="Run name (results folder)")
    parser.add_argument("--epochs", type=int, default=150, help="Maximum number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1376, help="Image size (must be multiple of 32)")
    parser.add_argument("--device", default=0, help="GPU device ID or 'cpu'")
    parser.add_argument("--workers", type=int, default=0, help="Number of dataloader workers (0 for Windows)")
    parser.add_argument("--optimizer", default="auto", help="Optimizer name, e.g. auto, AdamW, SGD")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience in epochs")
    parser.add_argument("--degrees", type=float, default=180.0, help="Random rotation range in degrees")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability")
    parser.add_argument("--flipud", type=float, default=0.5, help="Vertical flip probability")
    parser.add_argument("--seed", type=int, default=42, help="Random training seed")
    parser.add_argument("--single-cls", action="store_true", help="Train as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        degrees=args.degrees,
        fliplr=args.fliplr,
        flipud=args.flipud,
        seed=args.seed,
        save=True,
        project=args.project,
        name=args.name,
        pretrained=True,
        augment=args.augment,
        single_cls=args.single_cls,
    )

    print("CUDA available:", torch.cuda.is_available())


if __name__ == "__main__":
    main()
