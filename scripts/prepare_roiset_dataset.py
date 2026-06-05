"""Curate ImageJ ROI data and merge accepted images into a YOLO segmentation dataset."""

import argparse
import csv
import json
import random
import shutil
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image, ImageTk

from roi_utils import discover_image_roi_pairs, load_pair, polygon_mask, read_gray_image, read_roi_polygons


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


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
    """Multiply image brightness by factor while preserving dimensions and dtype."""
    factor = max(float(factor), 1.0)
    if np.issubdtype(image.dtype, np.integer):
        max_value = np.iinfo(image.dtype).max
        return np.clip(image.astype(np.float32) * factor, 0, max_value).astype(image.dtype)
    return np.clip(image.astype(np.float32) * factor, 0, 1.0)


def display_image(image: np.ndarray, brighten: bool, factor: float) -> Image.Image:
    image_to_show = brighten_image(image, factor) if brighten else image
    if image_to_show.ndim == 2:
        return Image.fromarray(normalize_to_uint8(image_to_show)).convert("RGB")
    if image_to_show.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(image_to_show, cv2.COLOR_BGRA2RGBA)).convert("RGB")
    if image_to_show.dtype != np.uint8:
        image_to_show = cv2.merge([normalize_to_uint8(channel) for channel in cv2.split(image_to_show)])
    return Image.fromarray(cv2.cvtColor(image_to_show, cv2.COLOR_BGR2RGB))


def load_json(path: Path, input_dir: Path) -> dict:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("images"), dict):
            raise ValueError(f"Invalid curation file: {path}")
        return data
    return {"version": 1, "input_dir": str(input_dir), "images": {}}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def dataset_root(dataset_yaml: Path) -> Tuple[Path, dict]:
    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = Path(config["path"])
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    return root, config


def split_image_dir(root: Path, config: dict, split: str) -> Path:
    path = Path(config[split])
    return path if path.is_absolute() else root / path


def yolo_line(points: np.ndarray, width: int, height: int) -> str:
    normalized = points.copy()
    normalized[:, 0] /= width
    normalized[:, 1] /= height
    normalized = np.clip(normalized, 0.0, 1.0)
    return "0 " + " ".join(f"{value:.6f}" for value in normalized.flatten())


def label_for_image(image_path: Path, split: str, root: Path) -> Path:
    return root / "labels" / split / f"{image_path.stem}.txt"


def copy_base_dataset(base_yaml: Path, output_dir: Path, input_stems: set[str]) -> dict:
    base_root, config = dataset_root(base_yaml)
    counts = {"train": 0, "val": 0}
    for split in ("train", "val"):
        source_images = split_image_dir(base_root, config, split)
        if not source_images.is_dir():
            continue
        target_images = output_dir / "images" / split
        target_labels = output_dir / "labels" / split
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(path for path in source_images.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
            if image_path.stem in input_stems:
                continue
            source_label = label_for_image(image_path, split, base_root)
            if not source_label.exists():
                raise FileNotFoundError(f"Missing label for base image: {source_label}")
            shutil.copy2(image_path, target_images / image_path.name)
            shutil.copy2(source_label, target_labels / source_label.name)
            counts[split] += 1
    return counts


def save_training_image(source: Path, target: Path, brighten: bool, factor: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not brighten:
        shutil.copy2(source, target)
        return
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image for brightness enhancement: {source}")
    brightened = brighten_image(image, factor)
    if not cv2.imwrite(str(target), brightened):
        raise ValueError(f"Cannot write brightened image: {target}")


def build_dataset(
    input_dir: Path,
    curation_path: Path,
    base_yaml: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
) -> dict:
    pairs = discover_image_roi_pairs(input_dir)
    curation = load_json(curation_path, input_dir)
    input_stems = {image_path.stem for image_path, _ in pairs}
    temp_output = output_dir.with_name(f"{output_dir.name}.__tmp_build")
    if temp_output.exists():
        shutil.rmtree(temp_output)
    if base_yaml.exists():
        base_counts = copy_base_dataset(base_yaml, temp_output, input_stems)
    else:
        for split in ("train", "val"):
            (temp_output / "images" / split).mkdir(parents=True, exist_ok=True)
            (temp_output / "labels" / split).mkdir(parents=True, exist_ok=True)
        base_counts = {"train": 0, "val": 0}

    accepted = []
    skipped = []
    for image_path, zip_path in pairs:
        entry = curation["images"].get(image_path.name, {})
        if not entry.get("completed") or entry.get("decision") != "keep":
            skipped.append({"image_file": image_path.name, "reason": entry.get("decision", "pending")})
            continue
        accepted.append((image_path, zip_path, entry))

    rng = random.Random(seed)
    accepted_names = [image_path.name for image_path, _, _ in accepted]
    rng.shuffle(accepted_names)
    val_count = int(round(len(accepted_names) * val_ratio)) if accepted_names else 0
    if accepted_names and val_ratio > 0:
        val_count = max(1, val_count)
    val_names = set(accepted_names[:val_count])

    backgrounds = {"version": 1, "source_curation": str(curation_path), "images": {}}
    validation_rows = []
    brightness_rows = []
    new_counts = {"train": 0, "val": 0, "worms": 0, "blank": 0}

    for image_path, zip_path, entry in accepted:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]
        gray = read_gray_image(image_path)
        polygons = read_roi_polygons(zip_path, width, height)
        roi_by_id = {polygon["id"]: polygon for polygon in polygons}
        blank_id = entry.get("blank_roi")
        if blank_id not in roi_by_id:
            raise ValueError(f"{image_path.name}: selected blank ROI no longer exists")

        split = "val" if image_path.name in val_names else "train"
        target_image = temp_output / "images" / split / image_path.name
        target_label = temp_output / "labels" / split / f"{image_path.stem}.txt"
        use_brightness = bool(entry.get("enhance_brightness", entry.get("enhance_contrast", False)))
        brightness_factor = float(entry.get("brightness_factor", 1.5))
        save_training_image(image_path, target_image, use_brightness, brightness_factor)

        label_lines = []
        seen = set()
        worm_mask = np.zeros(gray.shape, dtype=bool)
        for polygon in polygons:
            if polygon["id"] == blank_id:
                continue
            line = yolo_line(polygon["points"], width, height)
            if line in seen:
                continue
            seen.add(line)
            label_lines.append(line)
            worm_mask |= polygon_mask(gray.shape, polygon["points"])
        target_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        blank_polygon = roi_by_id[blank_id]
        blank_mask = polygon_mask(gray.shape, blank_polygon["points"])
        blank_pixels = gray[blank_mask]
        background_pixels = gray[~worm_mask]
        blank_mean = float(np.mean(blank_pixels)) if blank_pixels.size else None
        background_median = float(np.median(background_pixels)) if background_pixels.size else None
        validation_rows.append(
            {
                "image_file": image_path.name,
                "split": split,
                "worm_roi_count": len(label_lines),
                "blank_roi": blank_id,
                "blank_mean_intensity": blank_mean,
                "non_worm_background_median": background_median,
                "difference_blank_mean_minus_background_median": (
                    blank_mean - background_median
                    if blank_mean is not None and background_median is not None
                    else None
                ),
            }
        )
        brightness_rows.append(
            {
                "image_file": image_path.name,
                "split": split,
                "enhance_brightness": use_brightness,
                "brightness_factor": brightness_factor if use_brightness else 1.0,
                "training_image_file": target_image.name,
            }
        )
        backgrounds["images"][image_path.name] = {
            "zip_file": zip_path.name,
            "split": split,
            "blank_roi": blank_id,
            "blank_roi_name": blank_polygon["name"],
            "points": blank_polygon["points"].tolist(),
        }
        new_counts[split] += 1
        new_counts["worms"] += len(label_lines)
        new_counts["blank"] += 1

    yaml.safe_dump(
        {
            "path": temp_output.resolve().as_posix(),
            "train": "images/train",
            "val": "images/val",
            "test": None,
            "names": {0: "worm"},
        },
        (temp_output / "dataset.yaml").open("w", encoding="utf-8"),
        allow_unicode=True,
        sort_keys=False,
    )
    if validation_rows:
        with (temp_output / "background_validation.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(validation_rows[0]))
            writer.writeheader()
            writer.writerows(validation_rows)
    if brightness_rows:
        with (temp_output / "brightness_decisions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(brightness_rows[0]))
            writer.writeheader()
            writer.writerows(brightness_rows)
    (temp_output / "background_rois.json").write_text(
        json.dumps(backgrounds, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "input_dir": str(input_dir),
        "base_data": str(base_yaml),
        "output_dir": str(output_dir),
        "seed": seed,
        "val_ratio": val_ratio,
        "source_pairs": len(pairs),
        "accepted_new_images": len(accepted),
        "skipped_new_images": skipped,
        "base_counts": base_counts,
        "new_counts": new_counts,
        "final_train_images": len(list((temp_output / "images" / "train").iterdir())),
        "final_val_images": len(list((temp_output / "images" / "val").iterdir())),
    }
    (temp_output / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    temp_output.rename(output_dir)
    # Rewrite dataset.yaml after the final rename so the absolute path is correct.
    yaml.safe_dump(
        {
            "path": output_dir.resolve().as_posix(),
            "train": "images/train",
            "val": "images/val",
            "test": None,
            "names": {0: "worm"},
        },
        (output_dir / "dataset.yaml").open("w", encoding="utf-8"),
        allow_unicode=True,
        sort_keys=False,
    )
    return summary


class CurationApp:
    def __init__(
        self,
        root: tk.Tk,
        input_dir: Path,
        curation_path: Path,
        base_yaml: Path,
        output_dir: Path,
        val_ratio: float,
        seed: int,
    ) -> None:
        self.root = root
        self.input_dir = input_dir
        self.curation_path = curation_path
        self.base_yaml = base_yaml
        self.output_dir = output_dir
        self.val_ratio = val_ratio
        self.seed = seed
        self.pairs = discover_image_roi_pairs(input_dir)
        self.data = load_json(curation_path, input_dir)
        self.index = self.first_unfinished_index()
        self.image: Optional[np.ndarray] = None
        self.rois: List[dict] = []
        self.blank_roi: Optional[str] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.decision_var = tk.StringVar(value="keep")
        self.brightness_var = tk.BooleanVar(value=False)
        self.brightness_factor_var = tk.DoubleVar(value=1.5)
        self.note_var = tk.StringVar(value="")
        self.build_ui()
        self.load_current()

    def first_unfinished_index(self) -> int:
        for index, (image_path, _) in enumerate(self.pairs):
            if not self.data["images"].get(image_path.name, {}).get("completed"):
                return index
        return 0

    def build_ui(self) -> None:
        self.root.title("ROI data curation")
        self.root.geometry("1320x900")
        self.root.minsize(1000, 720)
        header = ttk.Frame(self.root, padding=8)
        header.pack(fill=tk.X)
        self.status = ttk.Label(header, text="")
        self.status.pack(side=tk.LEFT)
        self.summary = ttk.Label(header, text="")
        self.summary.pack(side=tk.RIGHT)

        body = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        body.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(body, background="#111111", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.select_from_canvas)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())

        side = ttk.Frame(body, width=330, padding=(10, 0, 0, 0))
        side.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(side, text="ROI list: click one blank/background ROI").pack(anchor=tk.W)
        self.roi_list = tk.Listbox(side, width=42, height=28, exportselection=False)
        self.roi_list.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
        self.roi_list.bind("<<ListboxSelect>>", self.select_from_list)

        decision_box = ttk.LabelFrame(side, text="Image decision")
        decision_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Radiobutton(decision_box, text="Keep for training", variable=self.decision_var, value="keep").pack(anchor=tk.W)
        ttk.Radiobutton(decision_box, text="Drop bad annotation/image", variable=self.decision_var, value="drop").pack(anchor=tk.W)
        ttk.Checkbutton(
            decision_box,
            text="Enhance brightness for training image",
            variable=self.brightness_var,
            command=self.redraw,
        ).pack(anchor=tk.W, pady=(4, 0))
        brightness_frame = ttk.Frame(decision_box)
        brightness_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(brightness_frame, text="Brightness factor").pack(side=tk.LEFT)
        ttk.Scale(
            brightness_frame,
            from_=1.0,
            to=3.0,
            variable=self.brightness_factor_var,
            command=lambda _value: self.redraw(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 6))
        self.brightness_label = ttk.Label(brightness_frame, width=5)
        self.brightness_label.pack(side=tk.RIGHT)

        ttk.Label(side, text="Note").pack(anchor=tk.W)
        ttk.Entry(side, textvariable=self.note_var).pack(fill=tk.X, pady=(0, 8))

        nav = ttk.Frame(side)
        nav.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(nav, text="Previous", command=self.previous).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(nav, text="Next", command=self.next).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))
        ttk.Button(side, text="Next unfinished", command=self.next_unfinished).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(side, text="Confirm, save, and rebuild dataset", command=self.confirm).pack(fill=tk.X)

    def current_pair(self) -> Tuple[Path, Path]:
        return self.pairs[self.index]

    def load_current(self) -> None:
        image_path, zip_path = self.current_pair()
        self.image, self.rois = load_pair(image_path, zip_path)
        entry = self.data["images"].get(image_path.name, {})
        self.blank_roi = entry.get("blank_roi")
        if self.blank_roi not in {roi["id"] for roi in self.rois}:
            self.blank_roi = None
        self.decision_var.set(entry.get("decision", "keep"))
        self.brightness_var.set(bool(entry.get("enhance_brightness", entry.get("enhance_contrast", False))))
        self.brightness_factor_var.set(float(entry.get("brightness_factor", 1.5)))
        self.note_var.set(entry.get("note", ""))
        self.populate_list()
        self.redraw()
        completed = sum(bool(self.data["images"].get(path.name, {}).get("completed")) for path, _ in self.pairs)
        self.status.config(text=f"{self.index + 1}/{len(self.pairs)}  {image_path.name}  ROI={len(self.rois)}")
        self.summary.config(text=f"completed {completed}/{len(self.pairs)}")

    def populate_list(self) -> None:
        self.roi_list.delete(0, tk.END)
        selected = None
        for index, roi in enumerate(self.rois):
            label = "blank" if roi["id"] == self.blank_roi else "worm"
            self.roi_list.insert(tk.END, f"{index + 1:02d}  {label:<5}  {roi['name']}")
            if roi["id"] == self.blank_roi:
                selected = index
        if selected is not None:
            self.roi_list.selection_set(selected)
            self.roi_list.see(selected)

    def redraw(self) -> None:
        if self.image is None:
            return
        canvas_w = max(self.canvas.winfo_width(), 10)
        canvas_h = max(self.canvas.winfo_height(), 10)
        height, width = self.image.shape[:2]
        self.scale = max(min((canvas_w - 20) / width, (canvas_h - 20) / height), 0.01)
        render_w = max(int(round(width * self.scale)), 1)
        render_h = max(int(round(height * self.scale)), 1)
        self.offset_x = (canvas_w - render_w) // 2
        self.offset_y = (canvas_h - render_h) // 2
        self.brightness_label.config(text=f"{self.brightness_factor_var.get():.2f}x")
        display = display_image(
            self.image,
            self.brightness_var.get(),
            self.brightness_factor_var.get(),
        ).resize((render_w, render_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo, anchor=tk.NW)
        for index, roi in enumerate(self.rois, start=1):
            points = roi["points"] * self.scale + np.array([self.offset_x, self.offset_y])
            color = "#ff3b30" if roi["id"] == self.blank_roi else "#19d7e6"
            self.canvas.create_polygon(points.flatten().tolist(), fill="", outline=color, width=3 if roi["id"] == self.blank_roi else 2)
            x, y = points[0]
            self.canvas.create_text(x + 5, y + 5, text=str(index), anchor=tk.NW, fill=color, font=("Arial", 10, "bold"))

    def set_blank(self, index: int) -> None:
        self.blank_roi = self.rois[index]["id"]
        self.populate_list()
        self.redraw()

    def select_from_list(self, _event=None) -> None:
        selection = self.roi_list.curselection()
        if selection:
            self.set_blank(selection[0])

    def select_from_canvas(self, event) -> None:
        x = (event.x - self.offset_x) / self.scale
        y = (event.y - self.offset_y) / self.scale
        matches = []
        for index, roi in enumerate(self.rois):
            distance = cv2.pointPolygonTest(roi["points"].astype(np.float32), (float(x), float(y)), True)
            if distance >= 0:
                matches.append((distance, index))
        if matches:
            self.set_blank(max(matches)[1])

    def confirm(self) -> None:
        image_path, zip_path = self.current_pair()
        if self.decision_var.get() == "keep" and self.blank_roi is None:
            messagebox.showwarning("Blank ROI required", "Choose one blank/background ROI before keeping this image.")
            return
        self.data["images"][image_path.name] = {
            "image_file": image_path.name,
            "zip_file": zip_path.name,
            "completed": True,
            "decision": self.decision_var.get(),
            "enhance_brightness": bool(self.brightness_var.get()),
            "brightness_factor": float(self.brightness_factor_var.get()) if self.brightness_var.get() else 1.0,
            "blank_roi": self.blank_roi,
            "note": self.note_var.get().strip(),
            "rois": [
                {
                    "id": roi["id"],
                    "name": roi["name"],
                    "label": "blank" if roi["id"] == self.blank_roi else "worm",
                }
                for roi in self.rois
            ],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(self.curation_path, self.data)
        try:
            build_dataset(self.input_dir, self.curation_path, self.base_yaml, self.output_dir, self.val_ratio, self.seed)
        except Exception as exc:
            messagebox.showerror("Dataset rebuild failed", str(exc))
            raise
        self.next_unfinished()

    def previous(self) -> None:
        self.index = (self.index - 1) % len(self.pairs)
        self.load_current()

    def next(self) -> None:
        self.index = (self.index + 1) % len(self.pairs)
        self.load_current()

    def next_unfinished(self) -> None:
        for offset in range(1, len(self.pairs) + 1):
            index = (self.index + offset) % len(self.pairs)
            image_name = self.pairs[index][0].name
            if not self.data["images"].get(image_name, {}).get("completed"):
                self.index = index
                self.load_current()
                return
        messagebox.showinfo("Done", "All images are completed. You can still use Previous/Next to edit them.")
        self.load_current()


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate ROI data and merge it into a YOLO dataset.")
    parser.add_argument("--input-dir", default="RoiSet/20260604_new_data", help="Directory with same-name images and ROI zip files")
    parser.add_argument("--base-data", default="label/combined_yolo_dataset_reviewed/dataset.yaml", help="Base dataset.yaml to copy before adding curated data")
    parser.add_argument("--output-dir", default="label/combined_yolo_dataset_reviewed", help="Output YOLO dataset directory")
    parser.add_argument("--curation", default="RoiSet/20260604_new_data/curation.json", help="Curation state JSON")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio for accepted new images")
    parser.add_argument("--seed", type=int, default=42, help="Split seed for accepted new images")
    parser.add_argument("--build-only", action="store_true", help="Rebuild dataset from existing curation JSON without opening GUI")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    curation_path = Path(args.curation)
    base_yaml = Path(args.base_data)
    output_dir = Path(args.output_dir)
    if args.build_only:
        summary = build_dataset(input_dir, curation_path, base_yaml, output_dir, args.val_ratio, args.seed)
        print(
            f"Dataset rebuilt: accepted={summary['accepted_new_images']}, "
            f"train={summary['final_train_images']}, val={summary['final_val_images']}"
        )
        return

    root = tk.Tk()
    try:
        CurationApp(root, input_dir, curation_path, base_yaml, output_dir, args.val_ratio, args.seed)
    except Exception as exc:
        root.withdraw()
        messagebox.showerror("Cannot start curation tool", str(exc))
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
