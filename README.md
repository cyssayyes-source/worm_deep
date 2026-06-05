# 荧光线虫 YOLO 自动分割与强度统计

本项目用于自动分割荧光线虫并统计每条线虫的荧光强度。当前公开版本以 **YOLO26s-seg** 为主模型：输入显微镜荧光图像，输出线虫实例分割预览图，以及每条线虫的原始平均强度、背景中位数和扣背景后的平均强度。

GitHub 仓库只发布代码、推理示例和可公开模型权重；训练集、原始 ROI/Labelme 标注、训练运行目录和本地待处理图像不随仓库开源。

## 本次更新

2026-06-05 版本主要更新如下：

- 新增并公开 YOLO26s 最优权重：`models/worm_yolo26s_best.pt`
- 新增 YOLO26s 原版推理输出：`examples/outputs/yolo26s_original/`
- 新增 YOLO26s 亮度增强检测输出：`examples/outputs/yolo26s_brightness_enhanced/`
- 将项目脚本统一整理到 `scripts/` 目录
- 新增 `scripts/prepare_roiset_dataset.py`，支持 ROI 数据人工筛选、blank ROI 选择、亮度增强选择，并自动重建 YOLO 数据集
- 新增 `scripts/review_dataset_labels.py` 和 `scripts/build_reviewed_dataset.py`，支持对已生成 YOLO 数据集进行人工复核和二次预处理
- 新增 `scripts/train_yolo26s_seg.py` 和 `scripts/train_yolo26m_seg.py`，用于 YOLO26 系列分割模型训练
- 增强 `scripts/infer_yolo.py`，支持亮度增强检测、重叠 mask 去重、最大连通域保留，以及原图强度统计
- README 补全每个脚本的使用方法和参数含义

## 开源内容

```text
models/
  worm_yolo26s_best.pt                 YOLO26s-seg 线虫分割最优权重
  worm_yolo_best.pt                    旧版 YOLOv8n-seg 微调权重，保留用于对比

examples/
  images/
    fluorescent_worm_example.jpg       推理示例输入图
  outputs/
    yolo26s_original/                  YOLO26s 原图检测输出
    yolo26s_brightness_enhanced/       YOLO26s 亮度增强检测输出

scripts/
  *.py                                 数据准备、训练、推理和复核工具
```

以下内容默认不入库：`RoiSet/`、`label/`、`hard_mining/`、`output_root*/`、`runs/`、`waiting_images/`、`worm_yolo_results/`、预训练下载权重和缓存文件。

## 环境安装

建议使用 Python 3.9 或相近版本的独立环境：

```powershell
conda create -n worm_env python=3.9 -y
conda activate worm_env
python -m pip install -r requirements.txt
```

依赖说明：

| 依赖 | 用途 |
|---|---|
| `ultralytics>=8.4.52` | YOLO26 训练和推理 |
| `opencv-python` | 图像读取、mask 处理、可视化输出 |
| `numpy` | 数值计算 |
| `torch` / `torchvision` | YOLO 训练和推理后端 |
| `labelme` | Labelme 标注解析 |
| `Pillow` | GUI 图像显示 |
| `PyYAML` | YOLO `dataset.yaml` 读写 |
| `roifile==2024.9.15` | ImageJ ROI zip 读取 |

## 快速推理

使用公开的 YOLO26s 权重和示例图：

```powershell
python scripts/infer_yolo.py `
  --model models/worm_yolo26s_best.pt `
  --input-dir examples/images `
  --output-root runs/example_yolo26s `
  --conf 0.1 `
  --imgsz 1376 `
  --overlap-thres 0.85 `
  --keep-largest-component
```

如果图像较暗，可以只对检测输入做亮度增强。注意：强度统计仍使用原图灰度，不使用增强后的灰度。

```powershell
python scripts/infer_yolo.py `
  --model models/worm_yolo26s_best.pt `
  --input-dir examples/images `
  --output-root runs/example_yolo26s_bright `
  --conf 0.1 `
  --imgsz 1376 `
  --enhance-brightness `
  --brightness-factor 1.5 `
  --overlap-thres 0.85 `
  --keep-largest-component
```

每张图会输出：

| 文件 | 含义 |
|---|---|
| `segmentation_result.jpg` | 带线虫轮廓、编号和右侧强度摘要的预览图 |
| `intensity_stats.csv` | 每条线虫的结构化强度统计 |
| `intensity_stats.txt` | 便于人工阅读的文本摘要 |

CSV 关键字段：

| 字段 | 含义 |
|---|---|
| `image_file` | 输入图像文件名 |
| `status` | 检测状态 |
| `worm_id` | 线虫编号 |
| `confidence` | 模型置信度 |
| `detection_image` | 检测使用 `original` 或 `brightness_enhanced` |
| `intensity_source` | 强度统计来源，默认 `original` |
| `brightness_factor` | 检测图亮度倍数 |
| `raw_mean_intensity` | 线虫 mask 内原始平均灰度 |
| `background_median` | 全部预测线虫 mask 外像素的背景中位数 |
| `corrected_mean_intensity` | `raw_mean_intensity - background_median` |

## 推荐工作流

1. 用 `prepare_roiset_dataset.py` 或旧版转换工具把 ROI/Labelme 标注转成 YOLO segmentation 数据集。
2. 用 `review_dataset_labels.py` 复核已有 YOLO 标签，并标记是否需要亮度增强。
3. 用 `build_reviewed_dataset.py` 根据复核结果生成干净训练集。
4. 用 `train_yolo26s_seg.py` 训练 YOLO26s-seg。
5. 用 `mine_hard_examples.py` 筛选难例，必要时用 `review_hard_examples.py` 人工复核。
6. 用 `build_hard_finetune_dataset.py` 构建难例强化微调集。
7. 用 `infer_yolo.py` 批量推理并导出强度统计。

## 脚本总览

所有命令建议在项目根目录运行。

| 脚本 | 用途 |
|---|---|
| `scripts/infer_yolo.py` | 批量推理、mask 后处理、背景校正强度统计 |
| `scripts/train_yolo26s_seg.py` | YOLO26s-seg 训练入口 |
| `scripts/train_yolo26m_seg.py` | YOLO26m-seg 训练入口 |
| `scripts/train_yolo.py` | 通用 YOLO segmentation 训练入口 |
| `scripts/prepare_roiset_dataset.py` | 新 ROI 数据筛选、blank ROI 选择、亮度增强选择、数据集合并 |
| `scripts/review_dataset_labels.py` | GUI 复核已生成 YOLO 数据集标签 |
| `scripts/build_reviewed_dataset.py` | 根据复核结果构建过滤/亮度增强后的 YOLO 数据集 |
| `scripts/annotate_rois.py` | 旧版 blank ROI 选择工具 |
| `scripts/roiset2yolo.py` | 旧版 ImageJ ROI zip 转 YOLO segmentation |
| `scripts/labelme2yolo.py` | Labelme polygon JSON 转 YOLO segmentation |
| `scripts/labelme2dataset.py` | Labelme JSON 转普通 image/mask 数据集 |
| `scripts/merge_yolo_datasets.py` | 合并两个 YOLO segmentation 数据集 |
| `scripts/mine_hard_examples.py` | 逐图评估并筛选难例 |
| `scripts/review_hard_examples.py` | GUI 复核难例 |
| `scripts/build_hard_finetune_dataset.py` | 构建难例强化微调数据集 |
| `scripts/roi_utils.py` | ROI 读取和 mask 工具库，无命令行入口 |

## scripts/infer_yolo.py

用途：批量运行 YOLO segmentation 推理，输出分割图和每条线虫的背景校正荧光强度。

示例：

```powershell
python scripts/infer_yolo.py `
  --model models/worm_yolo26s_best.pt `
  --input-dir waiting_images `
  --output-root output_root_yolo26s `
  --conf 0.1 `
  --imgsz 1376 `
  --overlap-thres 0.85 `
  --keep-largest-component
```

参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--model` | `models/worm_yolo26s_best.pt` | YOLO 分割模型权重路径 |
| `--input-dir` | `waiting_images` | 待推理图像目录 |
| `--output-root` | `output_root` | 输出根目录，每张图会建立一个子目录 |
| `--conf` | `0.1` | 检测置信度阈值 |
| `--imgsz` | `1376` | 推理图像尺寸 |
| `--enhance-brightness` | `False` | 只对检测输入做亮度增强 |
| `--enhance-contrast` | `False` | `--enhance-brightness` 的兼容别名 |
| `--no-enhance-brightness` | `False` | 显式关闭亮度增强 |
| `--brightness-factor` | `1.5` | 亮度增强倍数 |
| `--overlap-thres` | `0.85` | 若两个 mask 的交集/较小面积超过该阈值，抑制低置信度重复 mask；设为 `1.0` 可关闭 |
| `--keep-largest-component` | `True` | 每个预测 mask 只保留最大连通域 |
| `--no-keep-largest-component` | `False` | 保留预测 mask 的所有连通域 |

## scripts/train_yolo26s_seg.py

用途：训练 YOLO26s-seg。适合当前项目的主模型，8GB 显存建议从 `batch=1` 开始。

示例：

```powershell
python scripts/train_yolo26s_seg.py `
  --weights models/pretrained/yolo26s-seg.pt `
  --data label/combined_yolo_dataset_reviewed_bright/dataset.yaml `
  --project worm_yolo_results `
  --name train_yolo26s_seg `
  --epochs 300 `
  --patience 80 `
  --batch 1 `
  --imgsz 1376
```

参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--weights` | `models/pretrained/yolo26s-seg.pt` | 预训练权重、本地权重或模型名；支持 `yolov26s-seg`、`yolo26s-seg` 别名 |
| `--data` | `label/combined_yolo_dataset_reviewed/dataset.yaml` | YOLO segmentation 数据集配置 |
| `--project` | `worm_yolo_results` | 训练输出根目录 |
| `--name` | `train_yolo26s_seg` | 本次训练运行名 |
| `--epochs` | `300` | 最大训练轮数 |
| `--patience` | `80` | early stopping 等待轮数 |
| `--batch` | `1` | batch size |
| `--imgsz` | `1376` | 训练图像尺寸 |
| `--device` | `0` | GPU 编号或 `cpu` |
| `--workers` | `0` | dataloader worker 数，Windows 建议 `0` |
| `--optimizer` | `auto` | 优化器；`auto` 由 Ultralytics 自动选择 |
| `--lr0` | `None` | 初始学习率；仅在 `--optimizer` 不是 `auto` 时传给训练器 |
| `--degrees` | `180.0` | 随机旋转角度范围 |
| `--fliplr` | `0.5` | 左右翻转概率 |
| `--flipud` | `0.5` | 上下翻转概率 |
| `--seed` | `42` | 随机种子 |
| `--single-cls` | `False` | 将数据集按单类训练 |
| `--exist-ok` | `False` | 允许复用或覆盖同名运行目录 |

## scripts/train_yolo26m_seg.py

用途：训练 YOLO26m-seg。模型更大，显存占用更高。

示例：

```powershell
python scripts/train_yolo26m_seg.py `
  --weights models/pretrained/yolo26m-seg.pt `
  --data label/hard_finetune_dataset/dataset.yaml `
  --project worm_yolo_results `
  --name train_yolo26m_seg `
  --batch 1 `
  --imgsz 1376
```

参数与 `train_yolo26s_seg.py` 基本一致，默认差异如下：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--weights` | `models/pretrained/yolo26m-seg.pt` | YOLO26m-seg 预训练权重或模型名；支持 `yolov26m-seg`、`yolo26m-seg` 别名 |
| `--data` | `label/hard_finetune_dataset/dataset.yaml` | 默认使用难例强化数据集 |
| `--name` | `train_yolo26m_seg` | 训练运行名 |
| `--epochs` | `150` | 最大训练轮数 |
| `--patience` | `50` | early stopping 等待轮数 |

其他参数：`--project`、`--batch`、`--imgsz`、`--device`、`--workers`、`--optimizer`、`--lr0`、`--degrees`、`--fliplr`、`--flipud`、`--seed`、`--single-cls`、`--exist-ok`，含义同 YOLO26s 训练脚本。

## scripts/train_yolo.py

用途：通用 YOLO segmentation 训练入口，保留用于 YOLOv8 或其他权重的训练。

示例：

```powershell
python scripts/train_yolo.py `
  --weights models/worm_yolo26s_best.pt `
  --data label/combined_yolo_dataset/dataset.yaml `
  --project worm_yolo_results `
  --name train_custom
```

参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--weights` | `models/worm_yolo26s_best.pt` | 起始权重 |
| `--data` | `label/combined_yolo_dataset/dataset.yaml` | YOLO 数据集配置 |
| `--project` | `worm_yolo_results` | 训练输出根目录 |
| `--name` | `train_combined_finetune` | 训练运行名 |
| `--epochs` | `150` | 最大训练轮数 |
| `--batch` | `8` | batch size |
| `--imgsz` | `1376` | 图像尺寸 |
| `--device` | `0` | GPU 编号或 `cpu` |
| `--workers` | `0` | dataloader worker 数 |
| `--optimizer` | `auto` | 优化器 |
| `--lr0` | `0.001` | 初始学习率 |
| `--patience` | `50` | early stopping 等待轮数 |
| `--degrees` | `180.0` | 随机旋转范围 |
| `--fliplr` | `0.5` | 左右翻转概率 |
| `--flipud` | `0.5` | 上下翻转概率 |
| `--seed` | `42` | 随机种子 |
| `--single-cls` | `False` | 单类别训练 |
| `--augment` | `False` | 启用 Ultralytics 增强参数 |

## scripts/prepare_roiset_dataset.py

用途：推荐的新 ROI 数据入口。它会读取一组图像和同名 ImageJ ROI zip，打开 GUI 让用户选择 blank/background ROI、决定是否保留图像、是否对训练图做亮度增强，并自动合并到 YOLO 数据集。

示例：

```powershell
python scripts/prepare_roiset_dataset.py `
  --input-dir RoiSet/20260604_new_data `
  --base-data label/combined_yolo_dataset_reviewed/dataset.yaml `
  --output-dir label/combined_yolo_dataset_reviewed `
  --curation RoiSet/20260604_new_data/curation.json `
  --val-ratio 0.2 `
  --seed 42
```

参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--input-dir` | `RoiSet/20260604_new_data` | 包含图像和同名 ROI zip 的目录 |
| `--base-data` | `label/combined_yolo_dataset_reviewed/dataset.yaml` | 要先复制的基础 YOLO 数据集 |
| `--output-dir` | `label/combined_yolo_dataset_reviewed` | 合并后的输出数据集目录 |
| `--curation` | `RoiSet/20260604_new_data/curation.json` | GUI 筛选状态保存路径 |
| `--val-ratio` | `0.2` | 新接收图片进入验证集的比例 |
| `--seed` | `42` | 新图片 train/val 划分随机种子 |
| `--build-only` | `False` | 不打开 GUI，只根据已有 `curation.json` 重建数据集 |

GUI 操作：

| 控件 | 含义 |
|---|---|
| ROI list / 图像点击 | 选择唯一 blank/background ROI |
| `Keep for training` | 保留该图进入训练集 |
| `Drop bad annotation/image` | 丢弃该图 |
| `Enhance brightness for training image` | 写入训练集时保存亮度增强版 |
| `Brightness factor` | 亮度增强倍数 |
| `Confirm, save, and rebuild dataset` | 保存当前选择并重建输出数据集 |

输出文件：

| 文件 | 含义 |
|---|---|
| `dataset.yaml` | YOLO 训练入口 |
| `background_rois.json` | blank ROI 坐标记录 |
| `background_validation.csv` | blank ROI 与非线虫背景对比 |
| `brightness_decisions.csv` | 每张图是否亮度增强及倍数 |
| `conversion_summary.json` | 转换和合并统计 |

## scripts/review_dataset_labels.py

用途：GUI 复核已有 YOLO segmentation 数据集的标签质量，同时可标记输出数据集中是否需要亮度增强。

示例：

```powershell
python scripts/review_dataset_labels.py `
  --data label/combined_yolo_dataset_reviewed/dataset.yaml `
  --split all `
  --review-json label/combined_yolo_dataset_reviewed_preprocess_review.json `
  --review-csv label/combined_yolo_dataset_reviewed_preprocess_review.csv
```

参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--data` | `label/combined_yolo_dataset/dataset.yaml` | 要复核的 YOLO 数据集 |
| `--split` | `train` | 复核 `train`、`val` 或 `all` |
| `--review-json` | `label/label_review.json` | 复核状态 JSON |
| `--review-csv` | `label/label_review.csv` | 复核结果 CSV |
| `--max-width` | `1050` | GUI 预览最大宽度 |
| `--max-height` | `760` | GUI 预览最大高度 |

GUI 操作：`K` 保留、`D` 丢弃、`U` 标记不确定；也可选择是否在输出数据集中做亮度增强。

## scripts/build_reviewed_dataset.py

用途：根据 `review_dataset_labels.py` 产生的复核结果，构建过滤后的 YOLO 数据集；可同时把指定图像写成亮度增强版。

示例：

```powershell
python scripts/build_reviewed_dataset.py `
  --source-data label/combined_yolo_dataset_reviewed/dataset.yaml `
  --review-json label/combined_yolo_dataset_reviewed_preprocess_review.json `
  --output-dir label/combined_yolo_dataset_reviewed_bright `
  --reviewed-splits train val `
  --keep-pending `
  --overwrite
```

参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--source-data` | `label/combined_yolo_dataset/dataset.yaml` | 原始 YOLO 数据集 |
| `--review-json` | `label/label_review.json` | 复核 JSON |
| `--output-dir` | `label/combined_yolo_dataset_reviewed` | 输出数据集目录 |
| `--reviewed-splits` | `train` | 哪些 split 应按复核结果过滤，可传 `train val` |
| `--keep-unsure` | `False` | 保留标记为 unsure 的图 |
| `--keep-pending` | `False` | 保留未复核图 |
| `--overwrite` | `False` | 覆盖已有输出目录 |

## scripts/annotate_rois.py

用途：旧版工具，只负责为每张 ImageJ ROI 图选择一个 blank/background ROI，生成 `roi_labels.json`。

示例：

```powershell
python scripts/annotate_rois.py `
  --input-dir RoiSet `
  --annotations RoiSet/roi_labels.json
```

参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--input-dir` | `RoiSet` | 包含 TIFF 图像和同名 ROI zip 的目录 |
| `--annotations` | `RoiSet/roi_labels.json` | blank ROI 标注 JSON 输出路径 |

## scripts/roiset2yolo.py

用途：旧版 ImageJ ROI zip 转 YOLO segmentation 数据集。需要先用 `annotate_rois.py` 选择 blank ROI。

示例：

```powershell
python scripts/roiset2yolo.py `
  --input-dir RoiSet `
  --annotations RoiSet/roi_labels.json `
  --output-dir label/roi_yolo_dataset `
  --val-ratio 0.2 `
  --seed 42 `
  --overwrite
```

参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--input-dir` | `RoiSet` | 包含图像和 ROI zip 的目录 |
| `--annotations` | `RoiSet/roi_labels.json` | blank ROI 标注 JSON |
| `--output-dir` | `label/roi_yolo_dataset` | YOLO 数据集输出目录 |
| `--val-ratio` | `0.2` | 验证集比例 |
| `--seed` | `42` | train/val 划分随机种子 |
| `--overwrite` | `False` | 覆盖已有输出目录 |

## scripts/labelme2yolo.py

用途：将 Labelme polygon 标注转换为 YOLO segmentation 格式。

示例：

```powershell
python scripts/labelme2yolo.py `
  --labelme-dir label/converted `
  --yolo-dir label/yolo_dataset `
  --classes worm
```

参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--labelme-dir` | `label/converted` | Labelme JSON 和对应图像所在目录 |
| `--yolo-dir` | `label/yolo_dataset` | YOLO segmentation 输出目录 |
| `--classes` | `worm` | 要导出的类别名列表 |

## scripts/labelme2dataset.py

用途：将 Labelme JSON 转换为普通 `images/` + `masks/` 数据集，不是 YOLO segmentation 标签格式。

示例：

```powershell
python scripts/labelme2dataset.py `
  --labelme-dir label `
  --output-dir label/mask_dataset
```

参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--labelme-dir` | `label` | Labelme JSON 目录 |
| `--output-dir` | `label/yolo_dataset` | 输出 image/mask 数据集目录 |

## scripts/merge_yolo_datasets.py

用途：合并已有 train/val 划分的数据集和额外 YOLO 数据集。

示例：

```powershell
python scripts/merge_yolo_datasets.py `
  --base-dir label/roi_yolo_dataset `
  --extra-dir label/yolo_dataset `
  --output-dir label/combined_yolo_dataset `
  --extra-val-ratio 0.2 `
  --seed 42 `
  --overwrite
```

参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--base-dir` | `label/roi_yolo_dataset` | 已有 train/val 划分的基础数据集 |
| `--extra-dir` | `label/yolo_dataset` | 要加入的额外数据集 |
| `--output-dir` | `label/combined_yolo_dataset` | 合并输出目录 |
| `--extra-val-ratio` | `0.2` | 额外样本分入验证集的比例 |
| `--seed` | `42` | 额外样本划分随机种子 |
| `--overwrite` | `False` | 覆盖已有输出目录 |

## scripts/mine_hard_examples.py

用途：逐图评估模型表现，筛选低 mAP、低 recall 或漏检明显的难例。

示例：

```powershell
python scripts/mine_hard_examples.py `
  --model models/worm_yolo26s_best.pt `
  --data label/combined_yolo_dataset_reviewed_bright/dataset.yaml `
  --output-dir hard_mining/yolo26s_best `
  --map-threshold 0.4 `
  --recall-threshold 0.7 `
  --conf 0.001 `
  --detection-conf 0.1 `
  --imgsz 1376 `
  --device 0 `
  --save-previews
```

参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--model` | 必填 | 训练完成的 `best.pt` |
| `--data` | `label/combined_yolo_dataset/dataset.yaml` | YOLO 数据集 |
| `--output-dir` | `hard_mining/stage2_best` | 难例报告输出目录 |
| `--map-threshold` | `0.4` | 低于该 Mask mAP50-95 视为难例 |
| `--recall-threshold` | `0.7` | 低于该 Mask Recall@0.5 视为难例 |
| `--conf` | `0.001` | AP 计算用预测置信度阈值 |
| `--detection-conf` | `0.1` | 判定实际漏检用置信度阈值 |
| `--imgsz` | `1376` | 推理图像尺寸 |
| `--device` | `0` | 推理设备 |
| `--save-previews` | `False` | 保存难例预测预览图 |

## scripts/review_hard_examples.py

用途：GUI 复核 `mine_hard_examples.py` 筛出的难例，输出人工保留的难例清单。

示例：

```powershell
python scripts/review_hard_examples.py `
  --hard-report hard_mining/stage2_best/hard_examples.csv `
  --preview-dir hard_mining/stage2_best/previews `
  --review-json hard_mining/stage2_best/hard_review.json `
  --review-csv hard_mining/stage2_best/hard_review.csv `
  --kept-csv hard_mining/stage2_best/reviewed_hard_examples.csv
```

参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--hard-report` | `hard_mining/stage2_best/hard_examples.csv` | 难例 CSV |
| `--preview-dir` | `hard_mining/stage2_best/previews` | 难例预览图目录 |
| `--review-json` | `hard_mining/stage2_best/hard_review.json` | 复核状态 JSON |
| `--review-csv` | `hard_mining/stage2_best/hard_review.csv` | 全量复核结果 CSV |
| `--kept-csv` | `hard_mining/stage2_best/reviewed_hard_examples.csv` | 仅保留人工标记 keep 的难例 |
| `--max-width` | `1050` | GUI 预览最大宽度 |
| `--max-height` | `760` | GUI 预览最大高度 |

## scripts/build_hard_finetune_dataset.py

用途：根据难例报告复制原数据集，并对训练难例做重复采样和增强，构建强化微调数据集。

示例：

```powershell
python scripts/build_hard_finetune_dataset.py `
  --base-data label/combined_yolo_dataset/dataset.yaml `
  --hard-report hard_mining/stage2_best/hard_examples.csv `
  --output-dir label/hard_finetune_dataset `
  --repeat-copies 2 `
  --augment-copies 2 `
  --seed 42 `
  --overwrite
```

参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--base-data` | `label/combined_yolo_dataset/dataset.yaml` | 原始合并数据集 |
| `--hard-report` | `hard_mining/stage2_best/hard_examples.csv` | 难例筛选 CSV |
| `--output-dir` | `label/hard_finetune_dataset` | 强化训练数据集输出目录 |
| `--repeat-copies` | `2` | 每张训练难例原样重复次数 |
| `--augment-copies` | `2` | 每张训练难例额外生成增强图次数 |
| `--seed` | `42` | 亮度和噪声增强随机种子 |
| `--overwrite` | `False` | 覆盖已有输出目录 |

## scripts/roi_utils.py

用途：公共工具库，无命令行入口。主要负责发现图像/ROI zip 配对、读取 ImageJ ROI polygon、读取灰度图、生成 polygon mask 等。

## 许可证

本项目使用 MIT License 开源。
