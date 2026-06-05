"""Review YOLO segmentation labels and mark each image as keep/drop/unsure."""

import argparse
import csv
import json
import os
import subprocess
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image, ImageTk


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DECISION_KEEP = "keep"
DECISION_DROP = "drop"
DECISION_UNSURE = "unsure"
DECISION_PENDING = "pending"
DECISIONS = {DECISION_KEEP, DECISION_DROP, DECISION_UNSURE, DECISION_PENDING}
COLORS = [
    (0, 255, 0),
    (0, 180, 255),
    (255, 128, 0),
    (255, 0, 200),
    (0, 255, 255),
    (255, 80, 80),
    (160, 255, 0),
    (160, 80, 255),
]


def load_dataset(yaml_path: Path) -> Tuple[Path, dict]:
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(config["path"])
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root, config


def resolve_split_dir(root: Path, config: dict, split: str) -> Path:
    split_path = Path(config[split])
    return split_path if split_path.is_absolute() else root / split_path


def list_images(root: Path, config: dict, split: str) -> List[Path]:
    image_dir = resolve_split_dir(root, config, split)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def label_path(root: Path, split: str, image_path: Path) -> Path:
    return root / "labels" / split / f"{image_path.stem}.txt"


def item_key(split: str, image_path: Path) -> str:
    return f"{split}/{image_path.name}"


def load_reviews(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    reviews = data.get("reviews", data)
    clean = {}
    for key, value in reviews.items():
        decision = value.get("decision", DECISION_PENDING)
        clean[key] = {**value, "decision": decision if decision in DECISIONS else DECISION_PENDING}
    return clean


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image_f = image.astype(np.float32)
    min_value = float(np.min(image_f))
    max_value = float(np.max(image_f))
    if max_value <= min_value:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    return np.clip((image_f - min_value) * 255.0 / (max_value - min_value), 0, 255).astype(np.uint8)


def brighten_image(image: np.ndarray, factor: float) -> np.ndarray:
    factor = max(float(factor), 1.0)
    if np.issubdtype(image.dtype, np.integer):
        max_value = np.iinfo(image.dtype).max
        return np.clip(image.astype(np.float32) * factor, 0, max_value).astype(image.dtype)
    return np.clip(image.astype(np.float32) * factor, 0, 1.0)


def read_preview_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    if image.ndim == 2:
        gray = normalize_to_uint8(image)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype != np.uint8:
        channels = cv2.split(image)
        image = cv2.merge([normalize_to_uint8(channel) for channel in channels])
    return image


def read_label_polygons(path: Path, width: int, height: int) -> List[np.ndarray]:
    if not path.exists():
        return []
    polygons = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) < 7 or (len(values) - 1) % 2:
            raise ValueError(f"Invalid YOLO segmentation label at {path}:{line_number}")
        points = np.asarray([float(value) for value in values[1:]], dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        polygons.append(np.rint(points).astype(np.int32))
    return polygons


def polygon_center(points: np.ndarray) -> Tuple[int, int]:
    moments = cv2.moments(points)
    if moments["m00"]:
        return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
    return int(points[:, 0].mean()), int(points[:, 1].mean())


def draw_labels(
    image_path: Path,
    label_path_: Path,
    enhance_brightness: bool = False,
    brightness_factor: float = 1.5,
) -> Tuple[np.ndarray, int]:
    image = read_preview_image(image_path)
    if enhance_brightness:
        image = brighten_image(image, brightness_factor)
    height, width = image.shape[:2]
    overlay = image.copy()
    polygons = read_label_polygons(label_path_, width, height)
    for index, points in enumerate(polygons, start=1):
        color = COLORS[(index - 1) % len(COLORS)]
        cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=2)
        cv2.fillPoly(overlay, [points], color=color)
        x, y = polygon_center(points)
        cv2.putText(overlay, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
    blended = cv2.addWeighted(overlay, 0.32, image, 0.68, 0)
    for index, points in enumerate(polygons, start=1):
        color = COLORS[(index - 1) % len(COLORS)]
        cv2.polylines(blended, [points], isClosed=True, color=color, thickness=2)
    return blended, len(polygons)


def counts(items: List[dict], reviews: Dict[str, dict]) -> dict:
    result = {DECISION_KEEP: 0, DECISION_DROP: 0, DECISION_UNSURE: 0, DECISION_PENDING: 0}
    for item in items:
        decision = reviews.get(item["key"], {}).get("decision", DECISION_PENDING)
        result[decision if decision in result else DECISION_PENDING] += 1
    return result


def save_outputs(review_json: Path, review_csv: Path, items: List[dict], reviews: Dict[str, dict], dataset: Path) -> None:
    review_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "items": len(items),
        "counts": counts(items, reviews),
        "reviews": reviews,
    }
    review_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with review_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "split",
            "image_file",
            "label_file",
            "decision",
            "enhance_brightness",
            "brightness_factor",
            "reviewed_at",
            "note",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            review = reviews.get(item["key"], {})
            writer.writerow(
                {
                    "split": item["split"],
                    "image_file": item["image"].name,
                    "label_file": item["label"].name,
                    "decision": review.get("decision", DECISION_PENDING),
                    "enhance_brightness": bool(review.get("enhance_brightness", False)),
                    "brightness_factor": review.get("brightness_factor", 1.0),
                    "reviewed_at": review.get("reviewed_at", ""),
                    "note": review.get("note", ""),
                }
            )


def open_path(path: Path) -> None:
    if not path.exists():
        messagebox.showwarning("文件不存在", str(path))
        return
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class LabelReviewer:
    def __init__(
        self,
        root_window: tk.Tk,
        items: List[dict],
        dataset_yaml: Path,
        review_json: Path,
        review_csv: Path,
        max_width: int,
        max_height: int,
    ) -> None:
        self.root = root_window
        self.items = items
        self.dataset_yaml = dataset_yaml
        self.review_json = review_json
        self.review_csv = review_csv
        self.max_width = max_width
        self.max_height = max_height
        self.reviews = load_reviews(review_json)
        self.index = self.first_pending_index()
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.label_count = 0
        self.brightness_var = tk.BooleanVar(value=False)
        self.brightness_factor_var = tk.DoubleVar(value=1.5)
        self.rendering = False

        self.root.title("训练集标注复审")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build_ui()
        self.render()

    def build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        self.title_var = tk.StringVar()
        ttk.Label(main, textvariable=self.title_var, font=("Microsoft YaHei UI", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        body = ttk.Frame(main)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(body, anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(body, padding=(12, 0, 0, 0), width=330)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)

        self.info_var = tk.StringVar()
        ttk.Label(side, textvariable=self.info_var, justify="left", wraplength=320).grid(row=0, column=0, sticky="nw")

        ttk.Label(side, text="备注").grid(row=1, column=0, sticky="w", pady=(18, 2))
        self.note_var = tk.StringVar()
        self.note_entry = ttk.Entry(side, textvariable=self.note_var)
        self.note_entry.grid(row=2, column=0, sticky="ew")
        self.note_entry.bind("<FocusOut>", lambda _event: self.save_note())

        decisions = ttk.Frame(side)
        decisions.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        ttk.Button(decisions, text="保留 (K)", command=lambda: self.set_decision(DECISION_KEEP)).grid(row=0, column=0, sticky="ew", pady=2)
        ttk.Button(decisions, text="剔除坏标注 (D)", command=lambda: self.set_decision(DECISION_DROP)).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(decisions, text="未定 (U)", command=lambda: self.set_decision(DECISION_UNSURE)).grid(row=2, column=0, sticky="ew", pady=2)
        decisions.columnconfigure(0, weight=1)

        brightness = ttk.LabelFrame(side, text="Brightness preprocessing")
        brightness.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        ttk.Checkbutton(
            brightness,
            text="Enhance brightness in output dataset",
            variable=self.brightness_var,
            command=self.on_brightness_change,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(brightness, text="Factor").grid(row=1, column=0, sticky="w")
        ttk.Scale(
            brightness,
            from_=1.0,
            to=3.0,
            variable=self.brightness_factor_var,
            command=lambda _value: self.on_brightness_change(),
        ).grid(row=1, column=1, sticky="ew", padx=(8, 0))
        self.brightness_label = ttk.Label(brightness, text="1.50x")
        self.brightness_label.grid(row=2, column=0, columnspan=2, sticky="e")
        brightness.columnconfigure(1, weight=1)

        nav = ttk.Frame(side)
        nav.grid(row=5, column=0, sticky="ew", pady=(14, 0))
        ttk.Button(nav, text="上一张", command=self.previous).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(nav, text="下一张", command=self.next).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Button(nav, text="下一个未处理", command=self.next_pending).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(nav, text="打开图片", command=lambda: open_path(self.current_item()["image"])).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(nav, text="打开标签", command=lambda: open_path(self.current_item()["label"])).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        nav.columnconfigure(0, weight=1)
        nav.columnconfigure(1, weight=1)

        self.status_var = tk.StringVar()
        ttk.Label(main, textvariable=self.status_var).grid(row=2, column=0, sticky="w", pady=(8, 0))

        self.root.bind("<Left>", lambda _event: self.previous())
        self.root.bind("<Right>", lambda _event: self.next())
        self.root.bind("k", lambda _event: self.set_decision(DECISION_KEEP))
        self.root.bind("K", lambda _event: self.set_decision(DECISION_KEEP))
        self.root.bind("d", lambda _event: self.set_decision(DECISION_DROP))
        self.root.bind("D", lambda _event: self.set_decision(DECISION_DROP))
        self.root.bind("u", lambda _event: self.set_decision(DECISION_UNSURE))
        self.root.bind("U", lambda _event: self.set_decision(DECISION_UNSURE))

    def current_item(self) -> dict:
        return self.items[self.index]

    def first_pending_index(self) -> int:
        for index, item in enumerate(self.items):
            if self.reviews.get(item["key"], {}).get("decision", DECISION_PENDING) == DECISION_PENDING:
                return index
        return 0

    def persist(self) -> None:
        save_outputs(self.review_json, self.review_csv, self.items, self.reviews, self.dataset_yaml)

    def render(self) -> None:
        self.rendering = True
        item = self.current_item()
        review = self.reviews.get(item["key"], {})
        decision = review.get("decision", DECISION_PENDING)
        self.brightness_var.set(bool(review.get("enhance_brightness", False)))
        self.brightness_factor_var.set(float(review.get("brightness_factor", 1.5)))
        all_counts = counts(self.items, self.reviews)
        self.title_var.set(f"{self.index + 1}/{len(self.items)}  {item['key']}  [{decision}]")
        self.status_var.set(
            f"保留 {all_counts[DECISION_KEEP]} | 剔除 {all_counts[DECISION_DROP]} | 未定 {all_counts[DECISION_UNSURE]} | 未处理 {all_counts[DECISION_PENDING]}"
        )
        self.note_var.set(review.get("note", ""))

        try:
            preview, self.label_count = draw_labels(
                item["image"],
                item["label"],
                self.brightness_var.get(),
                self.brightness_factor_var.get(),
            )
            preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(preview)
            image.thumbnail((self.max_width, self.max_height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo, text="")
        except Exception as exc:
            self.photo = None
            self.image_label.configure(image="", text=str(exc))
            self.label_count = -1
        self.brightness_label.config(text=f"{self.brightness_factor_var.get():.2f}x")

        self.info_var.set(
            "\n".join(
                [
                    f"split: {item['split']}",
                    f"image: {item['image'].name}",
                    f"label: {item['label'].name}",
                    f"标注数量: {self.label_count}",
                    "",
                    "判断标准：",
                    "保留：轮廓基本贴合线虫，可用于训练。",
                    "剔除：漏标、错标、轮廓明显错位、背景当线虫等。",
                    "未定：需要之后再看。",
                    "",
                    "快捷键：K 保留，D 剔除，U 未定，左右方向键翻页。",
                ]
            )
        )
        self.rendering = False

    def save_note(self) -> None:
        item = self.current_item()
        review = self.reviews.setdefault(item["key"], {"decision": DECISION_PENDING})
        review["note"] = self.note_var.get().strip()
        review["enhance_brightness"] = bool(self.brightness_var.get())
        review["brightness_factor"] = float(self.brightness_factor_var.get()) if self.brightness_var.get() else 1.0
        self.persist()

    def on_brightness_change(self) -> None:
        if self.rendering:
            return
        item = self.current_item()
        review = self.reviews.setdefault(item["key"], {"decision": DECISION_PENDING})
        review["enhance_brightness"] = bool(self.brightness_var.get())
        review["brightness_factor"] = float(self.brightness_factor_var.get()) if self.brightness_var.get() else 1.0
        self.persist()
        self.render()

    def set_decision(self, decision: str) -> None:
        item = self.current_item()
        self.reviews[item["key"]] = {
            **self.reviews.get(item["key"], {}),
            "decision": decision,
            "enhance_brightness": bool(self.brightness_var.get()),
            "brightness_factor": float(self.brightness_factor_var.get()) if self.brightness_var.get() else 1.0,
            "note": self.note_var.get().strip(),
            "reviewed_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.persist()
        if self.index < len(self.items) - 1:
            self.index += 1
        self.render()

    def previous(self) -> None:
        self.save_note()
        self.index = max(0, self.index - 1)
        self.render()

    def next(self) -> None:
        self.save_note()
        self.index = min(len(self.items) - 1, self.index + 1)
        self.render()

    def next_pending(self) -> None:
        self.save_note()
        start = self.index + 1
        indices = list(range(start, len(self.items))) + list(range(0, start))
        for index in indices:
            if self.reviews.get(self.items[index]["key"], {}).get("decision", DECISION_PENDING) == DECISION_PENDING:
                self.index = index
                self.render()
                return
        messagebox.showinfo("完成", "没有未处理图片。")

    def on_close(self) -> None:
        self.save_note()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Review YOLO segmentation training labels in a GUI.")
    parser.add_argument("--data", default="label/combined_yolo_dataset/dataset.yaml", help="YOLO dataset.yaml")
    parser.add_argument("--split", choices=["train", "val", "all"], default="train", help="Dataset split to review")
    parser.add_argument("--review-json", default="label/label_review.json", help="Saved review state")
    parser.add_argument("--review-csv", default="label/label_review.csv", help="Review decisions as CSV")
    parser.add_argument("--max-width", type=int, default=1050, help="Maximum preview width")
    parser.add_argument("--max-height", type=int, default=760, help="Maximum preview height")
    args = parser.parse_args()

    dataset_yaml = Path(args.data)
    root, config = load_dataset(dataset_yaml)
    splits = ["train", "val"] if args.split == "all" else [args.split]
    items = []
    for split in splits:
        for image_path in list_images(root, config, split):
            label = label_path(root, split, image_path)
            if not label.exists():
                raise FileNotFoundError(f"Missing label for {image_path.name}: {label}")
            items.append({"split": split, "image": image_path, "label": label, "key": item_key(split, image_path)})
    if not items:
        raise ValueError("No images to review.")

    root_window = tk.Tk()
    LabelReviewer(
        root_window,
        items,
        dataset_yaml,
        Path(args.review_json),
        Path(args.review_csv),
        args.max_width,
        args.max_height,
    )
    root_window.mainloop()


if __name__ == "__main__":
    main()
