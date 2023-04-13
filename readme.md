# PyTorch Implementation of "HoneyChart"

HoneyChart is a collection of machine learning models that generate realistic and semantically consistent honey charts for honeyfiles.
HCA achieves high semantic consistency by structuring conditional prediction pathways from the surrounding document text, to the captions and then to the underlying chart data.
HCA reduces computational requirements by utilizing multi-task learning and a novel multi-head design that learns multiple chart formats simultaneously.

### Requirements
This codebase was built using Python 3.7 (CUDA 11.6), PyTorch 1.12.0, Torchvision 0.13.0. Use the following script to build a virtual env and install required packages.

```
python3.7 -m venv $HOME/envs/dcg
source $HOME/envs/dcg/bin/activate
pip install -U pip

pip install torch==1.12.0 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

### Setup

#### Default Configuration
In /config/default.yaml, ensure that 
* exp_dir.home -- points to your experiment directory
* data.path.home -- points to your dataset directory  

All other configurations inherit properties from default.yaml.

#### Datasets
The dataset combines charts and captions from PubMedCentral. The chart data was originally from the ICPR 2020 chart detection competition.
* Download - [link](https://decoychart.s3.ap-southeast-2.amazonaws.com/document-chart-dataset.zip)

This is automatically downloaded from a S3 bucket.

### Training and evaluation 
The entry point is main.py which requires specification of mode=['train','eval'],  stage=['chart_text','continuous','seq', 'generate']. 

The following command works for a single GPU. 
```
python main.py -c <CONFIG> -s <STAGE> -m <MODE>
``` 

The codebase is built using Pytorch Distributed Package.
Depending on your setup, the following is suitable for multiple GPUs:

```
srun torchrun --nnodes=<NNODES> \
    --nproc_per_node=<TASKS_PER_NODE> \
    --max_restarts=3 \
    --rdzv_id=<ID> \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR \
    main.py -c <CONFIG> -s <STAGE> -d <DATASET>
```

See official guide on [Pytorch Distributed](https://pytorch.org/docs/stable/distributed.html) for more information.


### License
Copyright © __________________________. This work has been supported by __________________________. We are currently tracking the impact __________________________ funded research. If you have used this code/data in your project, please contact us at __________________________ to let us know.

### Citations
Please cite our paper, if use this codebase. Thank you.
