# Fluorescent Worm YOLO Segmentation

本项目用于自动分割荧光线虫，并统计每条线虫的荧光强度。流程基于 Ultralytics YOLO 分割模型，输入显微镜荧光图像，输出实例分割预览图、每条线虫的原始平均强度、背景中位数和扣背景后的平均强度。

## 功能

- 自动识别并分割荧光线虫实例
- 为每条线虫生成编号和轮廓预览
- 统计原始平均荧光强度
- 使用非线虫区域估计背景中位数
- 输出扣背景后的线虫荧光强度
- 支持批量处理 `.png`、`.jpg`、`.jpeg`、`.tif`、`.tiff` 图像

## 开源内容

仓库包含：

- 推理、训练和数据转换脚本
- 最后一版公开模型权重：`models/worm_yolo_best.pt`
- 一个推理示例：
  - 输入图像：`examples/images/fluorescent_worm_example.jpg`
  - 分割结果：`examples/outputs/fluorescent_worm_segmentation_result.jpg`
  - 强度统计：`examples/outputs/fluorescent_worm_intensity_stats.csv`

仓库不包含训练集、原始标注数据、训练过程输出目录和中间结果。这些内容已通过 `.gitignore` 排除。

## 安装

建议使用独立 Python 环境：

```bash
conda create -n worm_env python=3.9 -y
conda activate worm_env
python -m pip install -r requirements.txt
```

## 快速推理

使用仓库自带示例图：

```bash
python infer_yolo.py ^
  --model models/worm_yolo_best.pt ^
  --input-dir examples/images ^
  --output-root runs/example ^
  --conf 0.1 ^
  --imgsz 1376
```

在 PowerShell 中也可以写成一行：

```powershell
python infer_yolo.py --model models/worm_yolo_best.pt --input-dir examples/images --output-root runs/example --conf 0.1 --imgsz 1376
```

运行后，每张图会在输出目录下生成：

- `segmentation_result.jpg`：线虫实例分割可视化结果
- `intensity_stats.csv`：每条线虫的荧光强度统计表
- `intensity_stats.txt`：便于人工查看的文本摘要

## 输出字段

`intensity_stats.csv` 包含以下字段：

- `image_file`：输入图像文件名
- `status`：检测状态
- `worm_id`：线虫编号
- `confidence`：模型置信度
- `raw_mean_intensity`：线虫区域原始平均灰度值
- `background_median`：非线虫区域背景中位数
- `corrected_mean_intensity`：扣背景后的平均荧光强度

## 训练

如果你有自己的 YOLO 分割数据集，可使用 `train_yolo.py` 训练或微调模型：

```bash
python train_yolo.py ^
  --weights models/worm_yolo_best.pt ^
  --data path/to/dataset.yaml ^
  --project worm_yolo_results ^
  --name train_custom ^
  --epochs 150 ^
  --batch 8 ^
  --imgsz 1376
```

训练数据需要符合 Ultralytics YOLO segmentation 数据集格式。当前仓库不发布原始训练集。

## 数据转换脚本

仓库保留了训练数据准备脚本，便于使用者在自己的数据上复现流程：

- `labelme2yolo.py`：将 Labelme 标注转换为 YOLO segmentation 格式
- `roiset2yolo.py`：将 ImageJ ROI zip 与图像转换为 YOLO segmentation 格式
- `merge_yolo_datasets.py`：合并多个 YOLO 数据集
- `build_hard_finetune_dataset.py`：构建 hard example 微调数据集
- `mine_hard_examples.py`：从验证结果中挖掘困难样本
- `annotate_rois.py`：辅助标注 ROI 中的背景区域

## 许可证

本项目使用 MIT License 开源。
