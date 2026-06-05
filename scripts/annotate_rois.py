"""Interactive selection of one blank/background ROI per ImageJ ROI set."""

import argparse
import json
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from roi_utils import discover_image_roi_pairs, load_pair


class RoiAnnotationApp:
    def __init__(self, root: tk.Tk, input_dir: Path, annotations_path: Path):
        self.root = root
        self.input_dir = input_dir
        self.annotations_path = annotations_path
        self.pairs = discover_image_roi_pairs(input_dir)
        self.data = self._read_annotations()
        self.index = self._first_unfinished_index()
        self.current_image = None
        self.current_rois = []
        self.current_blank = None
        self.photo = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self._build_ui()
        self._load_current()

    def _read_annotations(self) -> dict:
        if self.annotations_path.exists():
            with self.annotations_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("version") != 1 or not isinstance(data.get("images"), dict):
                raise ValueError(f"标注文件格式无效: {self.annotations_path}")
            return data
        return {"version": 1, "input_dir": str(self.input_dir), "images": {}}

    def _first_unfinished_index(self) -> int:
        for index, (image_path, _) in enumerate(self.pairs):
            entry = self.data["images"].get(image_path.name, {})
            if not entry.get("completed", False):
                return index
        return 0

    def _build_ui(self) -> None:
        self.root.title("Worm ROI 标注 - 点击一个 blank 背景圈")
        self.root.geometry("1280x860")
        self.root.minsize(980, 680)

        header = ttk.Frame(self.root, padding=8)
        header.pack(fill=tk.X)
        self.status = ttk.Label(header, text="")
        self.status.pack(side=tk.LEFT)
        self.hint = ttk.Label(header, text="点击图中轮廓或右侧列表选择唯一 blank；青色=worm，红色=blank")
        self.hint.pack(side=tk.RIGHT)

        content = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        content.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(content, background="#101010", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._select_from_canvas)
        self.canvas.bind("<Configure>", lambda _event: self._redraw())

        side = ttk.Frame(content, width=280, padding=(10, 0, 0, 0))
        side.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(side, text="ROI 列表（选择 blank）").pack(anchor=tk.W)
        self.roi_list = tk.Listbox(side, width=38, height=30, exportselection=False)
        self.roi_list.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
        self.roi_list.bind("<<ListboxSelect>>", self._select_from_list)

        nav = ttk.Frame(side)
        nav.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(nav, text="上一张", command=self._previous).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(nav, text="下一张", command=self._next).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))
        ttk.Button(side, text="跳至下一张未完成", command=self._next_unfinished).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(side, text="确认并保存本图", command=self._confirm).pack(fill=tk.X)

    def _load_current(self) -> None:
        image_path, zip_path = self.pairs[self.index]
        self.current_image, self.current_rois = load_pair(image_path, zip_path)
        entry = self.data["images"].get(image_path.name, {})
        self.current_blank = entry.get("blank_roi")
        if self.current_blank not in {roi["id"] for roi in self.current_rois}:
            self.current_blank = None
        self._populate_list()
        self._redraw()
        complete_count = sum(
            bool(self.data["images"].get(path.name, {}).get("completed")) for path, _ in self.pairs
        )
        suffix = "已确认" if entry.get("completed") else "未确认"
        self.status.config(
            text=f"{self.index + 1}/{len(self.pairs)}  {image_path.name}  "
            f"ROI={len(self.current_rois)}  [{suffix}]  总进度={complete_count}/{len(self.pairs)}"
        )

    def _populate_list(self) -> None:
        self.roi_list.delete(0, tk.END)
        selected_index = None
        for index, roi in enumerate(self.current_rois):
            label = "blank" if roi["id"] == self.current_blank else "worm"
            self.roi_list.insert(tk.END, f"{index + 1:02d}  {label:<5}  {roi['name']}")
            if label == "blank":
                selected_index = index
        if selected_index is not None:
            self.roi_list.selection_set(selected_index)
            self.roi_list.see(selected_index)

    def _display_image(self) -> Image.Image:
        image = self.current_image
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        if image.shape[2] == 4:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)).convert("RGB")
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def _redraw(self) -> None:
        if self.current_image is None:
            return
        canvas_width = max(self.canvas.winfo_width(), 10)
        canvas_height = max(self.canvas.winfo_height(), 10)
        height, width = self.current_image.shape[:2]
        self.scale = min((canvas_width - 20) / width, (canvas_height - 20) / height)
        self.scale = max(self.scale, 0.01)
        render_width = max(int(round(width * self.scale)), 1)
        render_height = max(int(round(height * self.scale)), 1)
        self.offset_x = (canvas_width - render_width) // 2
        self.offset_y = (canvas_height - render_height) // 2

        display = self._display_image().resize((render_width, render_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo, anchor=tk.NW)

        for index, roi in enumerate(self.current_rois, start=1):
            points = roi["points"] * self.scale + np.array([self.offset_x, self.offset_y])
            flattened = points.flatten().tolist()
            is_blank = roi["id"] == self.current_blank
            color = "#ff3b30" if is_blank else "#19d7e6"
            self.canvas.create_polygon(flattened, fill="", outline=color, width=3 if is_blank else 2)
            x, y = points[0]
            self.canvas.create_text(
                x + 5, y + 5, text=str(index), anchor=tk.NW, fill=color, font=("Arial", 10, "bold")
            )

    def _set_blank(self, index: int) -> None:
        self.current_blank = self.current_rois[index]["id"]
        self._populate_list()
        self._redraw()

    def _select_from_list(self, _event=None) -> None:
        selection = self.roi_list.curselection()
        if selection:
            self._set_blank(selection[0])

    def _select_from_canvas(self, event) -> None:
        if self.current_image is None:
            return
        x = (event.x - self.offset_x) / self.scale
        y = (event.y - self.offset_y) / self.scale
        matches = []
        for index, roi in enumerate(self.current_rois):
            contour = roi["points"].astype(np.float32)
            distance = cv2.pointPolygonTest(contour, (float(x), float(y)), True)
            if distance >= 0:
                matches.append((distance, index))
        if matches:
            self._set_blank(max(matches)[1])

    def _confirm(self) -> None:
        if self.current_blank is None:
            messagebox.showwarning("未选择 blank", "请先点击本图唯一的背景 ROI，再确认保存。")
            return
        image_path, zip_path = self.pairs[self.index]
        height, width = self.current_image.shape[:2]
        self.data["images"][image_path.name] = {
            "image_file": image_path.name,
            "zip_file": zip_path.name,
            "width": width,
            "height": height,
            "completed": True,
            "blank_roi": self.current_blank,
            "rois": [
                {
                    "id": roi["id"],
                    "name": roi["name"],
                    "label": "blank" if roi["id"] == self.current_blank else "worm",
                }
                for roi in self.current_rois
            ],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        with self.annotations_path.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)
        self._next_unfinished()

    def _previous(self) -> None:
        self.index = (self.index - 1) % len(self.pairs)
        self._load_current()

    def _next(self) -> None:
        self.index = (self.index + 1) % len(self.pairs)
        self._load_current()

    def _next_unfinished(self) -> None:
        for offset in range(1, len(self.pairs) + 1):
            index = (self.index + offset) % len(self.pairs)
            image_name = self.pairs[index][0].name
            if not self.data["images"].get(image_name, {}).get("completed"):
                self.index = index
                self._load_current()
                return
        messagebox.showinfo("标注完成", "所有图片都已确认。你仍可使用上一张/下一张返回修改。")
        self._load_current()


def main() -> None:
    parser = argparse.ArgumentParser(description="Select one blank/background ROI for each ImageJ ROI set.")
    parser.add_argument("--input-dir", default="RoiSet", help="包含 TIFF 图像与同名 ROI zip 的目录")
    parser.add_argument("--annotations", default="RoiSet/roi_labels.json", help="标注进度 JSON 输出路径")
    args = parser.parse_args()

    root = tk.Tk()
    try:
        RoiAnnotationApp(root, Path(args.input_dir), Path(args.annotations))
    except Exception as error:
        root.withdraw()
        messagebox.showerror("无法启动标注工具", str(error))
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
