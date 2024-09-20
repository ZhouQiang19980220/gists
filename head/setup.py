from setuptools import setup, find_packages

setup(
    name="head",
    version="0.1",
    py_modules=["head"],
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "torch",
        "torchvision",
        "tqdm",
        "Pillow",
    ],
    author="ZhouQiang",
    author_email="zhou_qiang98@163.com",
    description="A simple head file for common packages",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
