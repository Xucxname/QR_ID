from pathlib import Path

from setuptools import find_packages, setup


BASE_DIR = Path(__file__).resolve().parent

setup(
    name="qr-image-recognition",
    version="0.1.0",
    description="QR image recognition with OpenCV",
    long_description=(BASE_DIR / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "opencv-python",
        "numpy",
        "pillow",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "qr-id=qr_image_recognition.qr_id:main",
        ]
    },
)
