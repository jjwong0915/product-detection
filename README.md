# Product Detection Model

## Getting started
1. `poetry install` (installing tensorflow may take several minutes)
2. `poetry run python main.py`

## How to run with GPU
1. Install Docker 19.03, Nvidia GPU driver and Nvidia container toolkit
2. `docker run -it --rm --gpus all -v $(pwd):/app tensorflow/tensorflow:2.1.1-gpu bash`
