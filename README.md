# QR Image Recognition (Python)

项目已经开源至github：https://github.com/Xucxname/QR_ID (我的github仓库，后面也许还会更新)

基于图片进行二维码检测与解码的项目。输入单张图片，识别结果输出到命令行，并保存关键中间结果到 `output/` 目录。

## 功能

- 二维码图片信息识别
- 二维码信息加密及对称解密功能

## 环境

- Python: 3.10 / 3.11
- 所有需要的库：
- opencv-python
  numpy
  pillow
  qrcode
  cryptography
- 

## 安装依赖

```bash
pip install -e.
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

示例：

```bash
python scripts/QR_ID.py --name secret.png --gen-text "机密内容" --crypto fernet --key config/qr.key
python scripts/QR_ID.py --name secret.png --crypto fernet --key config/qr.key
```


### 生成密钥文件

```bash
python scripts/gen_key.py --name my.key
```

生成后的路径为 `config/my.key`。
`

## 输出说明

- 输出目录：`output/<图片名>/`
- 裁剪图：`crop.png`
- 预处理图：`crop_adapt.png`
- 逆透视图：`warp_adapt_1.png`、`warp_adapt_2.png`
- 可视化结果：`visualization.png`
- 文本结果：`result.txt`
