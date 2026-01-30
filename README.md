# QR Image Recognition (Python)

基于图片进行二维码检测与解码的项目。输入单张图片，识别结果输出到命令行，并保存关键中间结果到 `output/` 目录。

## 功能

- 单张图片识别二维码（命令行打印识别信息）
- 预处理仅保留自适应阈值（adapt），流程简洁
- 自动保存裁剪图、预处理图、逆透视图与可视化结果
- 基于 OpenCV `QRCodeDetector` 完成检测与解码

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
        ├── detect_core.py
        ├── detector.py
        ├── geometry.py
        ├── models.py
        ├── preprocess.py
        ├── qr_gen.py
        ├── qr_id.py
        └── visualize.py
```

## 使用方法

### 标准识别

```bash
python scripts/QR_ID.py --name your_image.png
```

### 生成二维码（保存到 images/）

```bash
python scripts/QR_ID.py --name new_qr.png --gen-text "hello world"
```

### 加密说明

- 支持：`none`、`fernet`、`xor`
- 通过 `--crypto` 选择方式，`--key` 指定密钥文件路径
- 未提供 `--key` 时默认明文（不加密/不解密）

示例：

```bash
python scripts/QR_ID.py --name secret.png --gen-text "机密内容" --crypto fernet --key config/qr.key
python scripts/QR_ID.py --name secret.png --crypto fernet --key config/qr.key
```

默认行为：

- 若 `config/qr.key` 存在，会自动读取并用于加/解密

### 生成密钥文件

```bash
python scripts/gen_key.py --name my.key
```

生成后的路径为 `config/my.key`。

## 识别流程概要

1. 全图检测二维码多边形，作为裁剪依据
2. 裁剪后生成 `adapt` 预处理图
3. 在裁剪图上再次检测并解码
4. 生成逆透视图（`warp_adapt_*.png`）用于检查

## 命令行输出示例

```
[OK] images/test_qr.png
  qr#1: https://example.com/?q=codex_qr_test&ts=2026-01-29
    polygon: [[49.0, 49.0], [442.0, 49.0], [442.95733642578125, 442.95733642578125], [49.0, 442.0]]
    box: [49.0, 49.0, 442.95733642578125, 442.95733642578125]
```

## 输出说明

- 输出目录：`output/<图片名>/`
- 裁剪图：`crop.png`
- 预处理图：`crop_adapt.png`
- 逆透视图：`warp_adapt_1.png`、`warp_adapt_2.png`
- 可视化结果：`visualization.png`
- 文本结果：`result.txt`
