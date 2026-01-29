# QR Image Recognition Project (Python)

基于图片进行二维码（QR Code）检测与解码的项目。给定单张或一批图片，识别结果会直接打印到命令行（非 JSON），可选输出可视化图片。

## 功能

- 单张图片识别二维码（命令行打印识别信息）
- 批量识别文件夹内图片（递归扫描）
- 可选保存可视化图（二维码边框叠加）
- 可选启用预处理增强（灰度/降噪/阈值/放大）

## 环境

- Python: 3.10 / 3.11
- 主要库：OpenCV（QRCodeDetector）

## 安装依赖

```bash
pip install -U pip
pip install -r requirements.txt
```

### 可编辑安装（避免设置 PYTHONPATH）

```bash
pip install -e .
```

## 项目结构

```
.
├── README.md
├── requirements.txt
└── src
    └── qr_image_recognition
        ├── __init__.py
        ├── cli.py
        ├── detector.py
        ├── image_io.py
        └── visualize.py
```

## 使用方法

### 单张图片识别（打印结果）

```bash
PYTHONPATH=src python -m qr_image_recognition.cli -i /path/to/image.png
```

### 使用 QR_ID 入口（images 文件夹）

```bash
PYTHONPATH=src python QR_ID.py --name your_image.png
```

### 批量识别文件夹

```bash
PYTHONPATH=src python -m qr_image_recognition.cli -f /path/to/images
```

### 启用预处理增强

```bash
PYTHONPATH=src python -m qr_image_recognition.cli -f /path/to/images --preprocess
```

### 仅检测定位（不解码）

```bash
PYTHONPATH=src python -m qr_image_recognition.cli -i /path/to/image.png --manual-detect --detect-only
```

### 手动检测 + OpenCV 解码

```bash
PYTHONPATH=src python -m qr_image_recognition.cli -i /path/to/image.png --manual-detect
```

### 保存可视化结果

```bash
PYTHONPATH=src python -m qr_image_recognition.cli -f /path/to/images --save-vis --vis-dir vis
```

### QR_ID 入口保存可视化

```bash
PYTHONPATH=src python QR_ID.py --name your_image.png --save-vis --vis-dir vis
```

### QR_ID 手动检测/仅检测

```bash
PYTHONPATH=src python QR_ID.py --name your_image.png --manual-detect
PYTHONPATH=src python QR_ID.py --name your_image.png --manual-detect --detect-only
```

## 命令行输出示例

```
[OK] /path/to/image.png
  qr#1: hello
    polygon: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    box: [x_min, y_min, x_max, y_max]
```
