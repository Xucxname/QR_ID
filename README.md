# QR Image Recognition Project (Python)

基于图片进行二维码（QR Code）检测与解码的项目骨架。目标是：给定单张或一批图片，输出二维码内容（text/data）、定位点（polygon/box），并可选保存可视化结果。

## 功能目标

- 单张图片识别二维码（返回文本 + 四边形角点）
- 批量识别文件夹内图片（递归扫描）
- 结果导出为 JSONL（每行一个图片记录，便于评测/分析）
- 可选保存可视化图（将二维码边框画在原图上）

## 推荐环境

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
└── src
    └── qr_image_recognition
        ├── __init__.py
        ├── cli.py
        ├── detector.py
        ├── image_io.py
        └── visualize.py
```

## 使用方法

### 单张图片识别

```bash
PYTHONPATH=src python -m qr_image_recognition.cli -i /path/to/image.png -o results.jsonl
```

### 批量识别文件夹

```bash
PYTHONPATH=src python -m qr_image_recognition.cli -f /path/to/images -o results.jsonl
```

### 保存可视化结果

```bash
PYTHONPATH=src python -m qr_image_recognition.cli -f /path/to/images -o results.jsonl --save-vis --vis-dir vis
```

### 启用预处理增强

```bash
PYTHONPATH=src python -m qr_image_recognition.cli -f /path/to/images -o results.jsonl --preprocess
```

## 输出格式（JSONL 每行一张图）

```json
{"image_path":"/path/to/image.png","found":true,"qrcodes":[{"text":"hello","polygon":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]],"box":[x_min,y_min,x_max,y_max]}],"error":null}
```

## 说明

- 默认使用 OpenCV 的 QRCodeDetector。
- 如果图片读入失败，会在 error 字段里记录原因。
