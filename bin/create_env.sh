#!/bin/bash

python3.7 -m venv $HOME/envs/dcg
source $HOME/envs/dcg/bin/activate
pip install -U pip

pip install torch==1.12.0 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt