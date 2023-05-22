# PyTorch Implementation of "HoneyPlotNet"

HoneyPlotNet is a deep learning architecture that generates realistic and semantically consistent charts for honeyfiles.
Our approach is to train a multimodal Transformer language model and multi-head vector quantization autoencoder to generate different components of a honeyplot based on the local document text and  caption.

### Software Requirements

This codebase was built using Python 3.7 and PyTorch 1.12.1. Use the following script to build a virtual env and install required packages.

```
python3.7 -m venv $HOME/envs/honeyplots
source $HOME/envs/honeyplots/bin/activate
pip install -U pip

# See https://pytorch.org/get-started/previous-versions/ 
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install -r requirements.txt
```

### Hardware Requirements
Training was conducted using a single A100 node with 4 GPUs (80GB). Each of the three stages required approximately 8-12 hours.

The trained weights for our best model (MVQGAN-T5) can be downloaded using `python download_weights.py`.

### Setup

#### Default Configuration
In /config/default.yaml, ensure that 
* `exp_dir.home` -- points to your experiment directory
* `data.path.home` -- points to your dataset directory  

All other configurations inherit properties from default.yaml.

#### Datasets
The dataset combines charts and captions from PubMedCentral. The chart data was originally from the ICPR 2020 chart detection competition.
This is automatically downloaded from a S3 bucket during training. It will be saved in your `data.path.home` config path.

### Training 
The entry point is main.py which requires specification of mode=['train','eval','generate],  stage=['continuous','seq']. 

`continuous` stage trains the Plot Data Model (PDM).

`seq` stage trains the multimodal Transformer with two decoders. Each decoder is trained seperately and is controled using the `model.seq.opt_mode` config setting. Each decoder must be trained in this order to replicate results.
* The first decoder is trained on language tokens (`model.seq.opt_mode: 1`). This freezes the weights of the second decoder.
* The second decoder is trained on data tokens (`model.seq.opt_mode: 2`). This freezes the shared encoder and language decoder.


The following command works for a single GPU. 
```
python main.py -c <CONFIG> -s <STAGE> -m <MODE>
``` 

The codebase is built using Pytorch Distributed Package.
Depending on your setup, the following is suitable for multiple GPUs:

```
torchrun --nnodes=<NNODES> \
    --nproc_per_node=<TASKS_PER_NODE> \
    --max_restarts=3 \
    --rdzv_id=<ID> \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR \
    main.py -c <CONFIG> -s <STAGE> -m <MODE>
```

See official guide on [Pytorch Distributed](https://pytorch.org/docs/stable/distributed.html) for more information.

### Evaluation

Once training is complete for all stages, you can conduct evaluation across all tasks in one run using eval.py.
```
python eval.py -c <CONFIG> 
``` 
Use the config file *mvqgan_t5.yaml* to replicate results in paper.

### License
Copyright © __________________________. This work has been supported by __________________________. 

### Citations
Please cite our paper, if use this codebase. Thank you.
