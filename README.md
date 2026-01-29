# QR Image Recognition Project (Python)

基于图片进行二维码（QR Code）检测与解码的项目。给定单张图片，识别结果会直接打印到命令行（非 JSON），并将每一步的结果保存到 output 文件夹。

## 功能

- 单张图片识别二维码（命令行打印识别信息）
- 默认启用预处理增强（灰度/降噪/阈值/放大）
- 自动保存每一步的处理结果与可视化图
- 只保留“手动检测 + OpenCV 解码”的流程

## 环境

- Python: 3.10 / 3.11
- 主要库：OpenCV（QRCodeDetector）

## 安装依赖

```bash
pip install -U pip
pip install -r requirements.txt
```

## 项目结构

```
.
├── README.md
├── requirements.txt
├── scripts
│   └── QR_ID.py
├── output
└── src
    └── qr_image_recognition
        ├── __init__.py
        ├── detector.py
        ├── image_io.py
        ├── qr_id.py
        └── visualize.py
```

## 使用方法

### 标准运行（脚本）

```bash
python scripts/QR_ID.py --name your_image.png
```

## 命令行输出示例

```
[OK] images/test_qr.png
  qr#1: https://example.com/?q=codex_qr_test&ts=2026-01-29
    polygon: [[49.0, 49.0], [442.0, 49.0], [442.95733642578125, 442.95733642578125], [49.0, 442.0]]
    box: [49.0, 49.0, 442.95733642578125, 442.95733642578125]
```

## 输出说明

- 输出目录：`output/<图片名>/`
- 包含预处理结果图：`original.png`、`gray.png`、`denoised.png`、`thresh.png`、`scaled.png`（视情况）
- 可视化结果：`visualization.png`
- 文本结果：`result.txt`
