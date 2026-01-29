# qr_image_project_all_in_one.md

下面是一份**把当前项目的所有关键文件（README + requirements + 源码）整理到同一个 Markdown 文档**的“合集版”。你可以直接把本文件保存为 `qr_image_project_all_in_one.md`，后续交给 Codex/自己照着生成对应文件即可。

---

## 1) README.md（项目说明 + 功能 + 使用方法）

# QR Image Recognition Project (Python)

基于**图片**进行二维码（QR Code）检测与解码的项目骨架。目标是：给定单张或一批图片，输出二维码内容（text/data）、定位点（polygon/box），并可选保存可视化结果。

---

## 功能目标

- ✅ 单张图片识别二维码（返回文本 + 四边形角点）
- ✅ 批量识别文件夹内图片（递归扫描）
- ✅ 结果导出为 `JSONL`（每行一个图片记录，便于评测/分析）
- ✅ 可选保存可视化图（将二维码边框画在原图上）
- 🔜（可选增强）图像预处理提升识别率（降噪/二值化/放大/矫正）
- 🔜（可选增强）OpenCV + pyzbar 双解码兜底（更稳但需 zbar）

---

## 推荐环境

- Python: 3.10 / 3.11
- 主要库：OpenCV（`QRCodeDetector`）

### 安装依赖（最小可用）
```bash
pip install -U pip
pip install opencv-python numpy pillow
