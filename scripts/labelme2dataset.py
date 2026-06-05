import argparse
import os
import json
import numpy as np
import cv2
from labelme import utils


def convert_labelme_to_dataset(labelme_dir, output_dir):
    """Convert a Labelme directory to a dataset of images + masks."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    for json_file in os.listdir(labelme_dir):
        if not json_file.endswith(".json"):
            continue

        json_path = os.path.join(labelme_dir, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img = utils.img_b64_to_arr(data["imageData"])
        lbl, _, _ = utils.labelme_shapes_to_label(img.shape, data["shapes"])

        img_path = os.path.join(output_dir, "images", json_file.replace(".json", ".png"))
        mask_path = os.path.join(output_dir, "masks", json_file.replace(".json", "_mask.png"))

        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, lbl)


def main():
    parser = argparse.ArgumentParser(description="Convert Labelme JSON annotations into image+mask dataset.")
    parser.add_argument("--labelme-dir", default="label", help="Labelme JSON directory")
    parser.add_argument("--output-dir", default="label/yolo_dataset", help="Output dataset directory")
    args = parser.parse_args()

    convert_labelme_to_dataset(args.labelme_dir, args.output_dir)
    print(" 转换完成！输出目录：", args.output_dir)


if __name__ == "__main__":
    main()
