"""Review mined hard examples in a small Tkinter UI."""

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
from typing import Dict, List, Optional

from PIL import Image, ImageTk


DECISION_KEEP = "keep"
DECISION_DROP = "drop"
DECISION_UNSURE = "unsure"
DECISION_PENDING = "pending"
DECISIONS = {DECISION_KEEP, DECISION_DROP, DECISION_UNSURE, DECISION_PENDING}


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict) -> str:
    return f"{row['split']}/{row['image_file']}"


def preview_name(row: dict) -> str:
    return f"{row['split']}__{Path(row['image_file']).stem}.jpg"


def load_reviews(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    reviews = data.get("reviews", data)
    result = {}
    for key, value in reviews.items():
        decision = value.get("decision", DECISION_PENDING)
        if decision not in DECISIONS:
            decision = DECISION_PENDING
        result[key] = {**value, "decision": decision}
    return result


def save_json(path: Path, rows: List[dict], reviews: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_images": len(rows),
        "counts": decision_counts(rows, reviews),
        "reviews": reviews,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_review_csv(path: Path, rows: List[dict], reviews: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = list(rows[0].keys()) if rows else []
    fieldnames = base_fields + ["review_decision", "reviewed_at", "review_note"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            review = reviews.get(row_key(row), {})
            writer.writerow(
                {
                    **row,
                    "review_decision": review.get("decision", DECISION_PENDING),
                    "reviewed_at": review.get("reviewed_at", ""),
                    "review_note": review.get("note", ""),
                }
            )


def write_kept_csv(path: Path, rows: List[dict], reviews: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = list(rows[0].keys()) if rows else []
    fieldnames = base_fields + ["review_decision", "reviewed_at", "review_note"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            review = reviews.get(row_key(row), {})
            if review.get("decision") != DECISION_KEEP:
                continue
            writer.writerow(
                {
                    **row,
                    "review_decision": DECISION_KEEP,
                    "reviewed_at": review.get("reviewed_at", ""),
                    "review_note": review.get("note", ""),
                }
            )


def decision_counts(rows: List[dict], reviews: Dict[str, dict]) -> dict:
    counts = {DECISION_KEEP: 0, DECISION_DROP: 0, DECISION_UNSURE: 0, DECISION_PENDING: 0}
    for row in rows:
        decision = reviews.get(row_key(row), {}).get("decision", DECISION_PENDING)
        counts[decision if decision in counts else DECISION_PENDING] += 1
    return counts


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


class HardExampleReviewer:
    def __init__(
        self,
        root: tk.Tk,
        rows: List[dict],
        preview_dir: Path,
        review_json: Path,
        review_csv: Path,
        kept_csv: Path,
        max_width: int,
        max_height: int,
    ) -> None:
        self.root = root
        self.rows = rows
        self.preview_dir = preview_dir
        self.review_json = review_json
        self.review_csv = review_csv
        self.kept_csv = kept_csv
        self.max_width = max_width
        self.max_height = max_height
        self.reviews = load_reviews(review_json)
        self.index = self.first_pending_index()
        self.photo: Optional[ImageTk.PhotoImage] = None

        self.root.title("坏例复审 - 选择是否保留")
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

        content = ttk.Frame(main)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(content, anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(content, padding=(12, 0, 0, 0), width=330)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)

        self.info_text = tk.Text(side, width=42, height=22, wrap="word")
        self.info_text.grid(row=0, column=0, sticky="nsew")
        self.info_text.configure(state="disabled")

        ttk.Label(side, text="备注").grid(row=1, column=0, sticky="w", pady=(10, 2))
        self.note_var = tk.StringVar()
        self.note_entry = ttk.Entry(side, textvariable=self.note_var)
        self.note_entry.grid(row=2, column=0, sticky="ew")
        self.note_entry.bind("<FocusOut>", lambda _event: self.save_current_note())

        decision_frame = ttk.Frame(side)
        decision_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(decision_frame, text="保留为坏例 (K)", command=lambda: self.set_decision(DECISION_KEEP)).grid(
            row=0, column=0, sticky="ew", pady=2
        )
        ttk.Button(decision_frame, text="剔除 (D)", command=lambda: self.set_decision(DECISION_DROP)).grid(
            row=1, column=0, sticky="ew", pady=2
        )
        ttk.Button(decision_frame, text="未定 (U)", command=lambda: self.set_decision(DECISION_UNSURE)).grid(
            row=2, column=0, sticky="ew", pady=2
        )
        decision_frame.columnconfigure(0, weight=1)

        nav_frame = ttk.Frame(side)
        nav_frame.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        ttk.Button(nav_frame, text="上一张", command=self.previous).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(nav_frame, text="下一张", command=self.next).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Button(nav_frame, text="下一个未处理", command=self.next_pending).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(nav_frame, text="打开预览图", command=self.open_current_preview).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        nav_frame.columnconfigure(0, weight=1)
        nav_frame.columnconfigure(1, weight=1)

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

    def current_row(self) -> dict:
        return self.rows[self.index]

    def current_key(self) -> str:
        return row_key(self.current_row())

    def current_preview_path(self) -> Path:
        return self.preview_dir / preview_name(self.current_row())

    def first_pending_index(self) -> int:
        for index, row in enumerate(self.rows):
            if self.reviews.get(row_key(row), {}).get("decision", DECISION_PENDING) == DECISION_PENDING:
                return index
        return 0

    def render(self) -> None:
        row = self.current_row()
        key = self.current_key()
        review = self.reviews.get(key, {})
        decision = review.get("decision", DECISION_PENDING)
        counts = decision_counts(self.rows, self.reviews)

        self.title_var.set(f"{self.index + 1}/{len(self.rows)}  {key}  [{decision}]")
        self.status_var.set(
            f"保留 {counts[DECISION_KEEP]} | 剔除 {counts[DECISION_DROP]} | 未定 {counts[DECISION_UNSURE]} | 未处理 {counts[DECISION_PENDING]}"
        )
        self.note_var.set(review.get("note", ""))

        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        info_lines = [
            f"split: {row.get('split', '')}",
            f"image: {row.get('image_file', '')}",
            f"reason: {row.get('reasons', '')}",
            "",
            f"GT count: {row.get('gt_count', '')}",
            f"Pred count: {row.get('pred_count', '')}",
            f"Detected @conf: {row.get('pred_count_at_detection_conf', '')}",
            f"Matched IoU50: {row.get('matched_mask_iou50', '')}",
            "",
            f"Mask AP50: {row.get('mask_ap50', '')}",
            f"Mask mAP50-95: {row.get('mask_map50_95', '')}",
            f"Mask recall50: {row.get('mask_recall50', '')}",
            f"Mean best GT IoU: {row.get('mean_best_gt_iou', '')}",
            "",
            "快捷键：K 保留，D 剔除，U 未定，左右方向键翻页。",
        ]
        self.info_text.insert("1.0", "\n".join(info_lines))
        self.info_text.configure(state="disabled")

        preview_path = self.current_preview_path()
        if preview_path.exists():
            image = Image.open(preview_path)
            image.thumbnail((self.max_width, self.max_height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo, text="")
        else:
            self.photo = None
            self.image_label.configure(image="", text=f"找不到预览图:\n{preview_path}")

    def persist(self) -> None:
        save_json(self.review_json, self.rows, self.reviews)
        write_review_csv(self.review_csv, self.rows, self.reviews)
        write_kept_csv(self.kept_csv, self.rows, self.reviews)

    def save_current_note(self) -> None:
        key = self.current_key()
        review = self.reviews.setdefault(key, {"decision": DECISION_PENDING})
        review["note"] = self.note_var.get().strip()
        self.persist()

    def set_decision(self, decision: str) -> None:
        key = self.current_key()
        self.reviews[key] = {
            **self.reviews.get(key, {}),
            "decision": decision,
            "note": self.note_var.get().strip(),
            "reviewed_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.persist()
        if self.index < len(self.rows) - 1:
            self.index += 1
        self.render()

    def previous(self) -> None:
        self.save_current_note()
        self.index = max(0, self.index - 1)
        self.render()

    def next(self) -> None:
        self.save_current_note()
        self.index = min(len(self.rows) - 1, self.index + 1)
        self.render()

    def next_pending(self) -> None:
        self.save_current_note()
        start = self.index + 1
        indices = list(range(start, len(self.rows))) + list(range(0, start))
        for index in indices:
            if self.reviews.get(row_key(self.rows[index]), {}).get("decision", DECISION_PENDING) == DECISION_PENDING:
                self.index = index
                self.render()
                return
        messagebox.showinfo("完成", "没有未处理的坏例。")

    def open_current_preview(self) -> None:
        open_path(self.current_preview_path())

    def on_close(self) -> None:
        self.save_current_note()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a GUI to review mined hard examples.")
    parser.add_argument("--hard-report", default="hard_mining/stage2_best/hard_examples.csv", help="hard_examples.csv")
    parser.add_argument("--preview-dir", default="hard_mining/stage2_best/previews", help="Directory containing preview JPGs")
    parser.add_argument("--review-json", default="hard_mining/stage2_best/hard_review.json", help="Saved review state")
    parser.add_argument("--review-csv", default="hard_mining/stage2_best/hard_review.csv", help="All review decisions")
    parser.add_argument(
        "--kept-csv",
        default="hard_mining/stage2_best/reviewed_hard_examples.csv",
        help="CSV containing only examples marked keep",
    )
    parser.add_argument("--max-width", type=int, default=1050, help="Maximum preview width")
    parser.add_argument("--max-height", type=int, default=760, help="Maximum preview height")
    args = parser.parse_args()

    rows = read_csv_rows(Path(args.hard_report))
    if not rows:
        raise ValueError(f"No rows found in {args.hard_report}")

    root = tk.Tk()
    HardExampleReviewer(
        root,
        rows,
        Path(args.preview_dir),
        Path(args.review_json),
        Path(args.review_csv),
        Path(args.kept_csv),
        args.max_width,
        args.max_height,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
