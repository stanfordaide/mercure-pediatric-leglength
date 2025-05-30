from setuptools import setup, find_packages

setup(
    name="fw_leglength",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.1",
        "torchvision>=0.17.1",
        "torchmetrics>=1.7.1",
        "numpy>=1.26.4",
        "pandas>=2.2.1",
        "pydicom>=2.4.4",
        "opencv-python>=4.10.0",
        "timm>=1.0.15",
        "tqdm>=4.67.1",
        "pycocotools>=2.0.8",
    ],
) 