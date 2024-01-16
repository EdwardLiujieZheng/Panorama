# Panorama Stitcher
This project creates panorama from a sequence of images.
<img width="663" alt="image" src="https://github.com/liujie-zheng/panorama-stitcher/assets/69744953/5f2dfe54-26e3-44f3-9955-5922fa6bfa20">

## Installation
1. Clone this repository by running:
```
git@github.com:liujie-zheng/panorama-stitcher.git
cd panorama-stitcher
```
2. Install conda [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
3. Install dependencies by runnning:
```
conda env create -f environment.yml
conda activate panorama
```

## Run a demo
Running on sample images:
```
python main.py
```
Running on custom images:
```
python main.py --input <input directory> --output <output directory> --crop <up> <down> <left> <right>
```
