# PyTorch Implementation of "Decoy Chart Generator for Cyber Deception"

<div align="center">
  <a href="http://dngu7.github.io" target="_blank">David&nbsp;D. Nguyen</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://au.linkedin.com/in/david-liebowitz-1945044" target="_blank">David&nbsp;Liebowitz</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://people.csiro.au/N/S/Surya-Nepal" target="_blank">Surya&nbsp;Nepal</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://salilkanhere.net/" target="_blank">Salil&nbsp;Kanhere</a> 
</div>
<br>
<br>


[Decoy Chart Generator] (DCG) is a combination of machine learning models that generates realistic and semantically consistency charts for honeyfiles.
DCG achieves high semantic consistency by structuring conditional prediction pathways from the surrounding document text, to the captions and then to the underlying chart data.
DCG reduces computational requirements by utilizing multi-task learning and a novel multi-head design that learns multiple chart types simultaneously.


### Requirements
This codebase was built using Python 3.7 (cuda10.2) and PyTorch 1.11.0. Run the following for additional packages.
```
pip install -r requirements.txt
``` 

### Setup

#### Default Configuration
In /config/default.yaml, ensure that 
* exp_dir.home -- points to your experiment directory
* data.path.home -- points to your dataset directory  

All configurations inherit properties from default.yaml (ensure they exist in default first.)

#### Datasets
The dataset combines charts and captions from PubMedCentral. The chart data was originally from the ICPR 2020 chart detection competition.
* Download - [link](https://decoychart.s3.ap-southeast-2.amazonaws.com/document-chart-dataset.zip)

Put this into your config.data.path.home directory.

#### Precompute KSM
Run the following command to precompute FID weights. FID statistics are saved in data directory.
```
python pre_fid.py --dataset <DATASET> --data_dir <DATA_DIR>
``` 

### Training and evaluation 
The entry point is main.py which requires specification of mode=['train','eval'],  stage=['vae','seq'] and dataset=['mnist', 'kmnist', 'fashion', 'cifar10', 'celeba64', 'imagenet32', 'imagenet64']. 
* stage=="vae" trains the codebook, encoder and decoder.
* stage=="seq" trains the prior model (pixel or gpt).

The following command works for a single GPU. 
```
python main.py -c <CONFIG> -s <STAGE> -d <DATASET>
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

#### Compute resources
* FashionMNIST - 1 P100 for ~48 hours for each stage.
* Celeba64 - 8 P100 for ~48 hours for each stage.
* Imagenet64 - 16 A100 for ~72 hours for each stage. Note that the requirements here are not suitable for A100. A different cuda version required.
See paper for more details.

#### Evaluation
Ensure fid weights have been precomputed as per above instructions.
The frequency of evaluation can be modified in the configuration file with "train.intervals.eval". 
Warmup length can also be modified with "train.epochs.warmup".

### Other Settings

##### Distributed dropout
A simple distributed dropout block (described in the paper) is implemented in /models/dist_dropout/ddb_2d.py.
This can be activated/deactivated in config with "model.ddb.use".

##### VQGAN
The discriminator can be activated in config with "model.gan.use". 
VQGAN encoder/encoder is also integrated with the "model.coder.name"=="conv_taming", however this has not been thoroughly tested yet.

##### VQVAE-2
The VQVAE2 backbone is activated with "model.backbone"=="vq2". Distributed dropout is not compatible with this option, thus will be ignored if true.

### Architecture
The following diagram provides an overview of the architecture.


### License
Copyright © __________________________ 2022. This work has been supported by the Cyber Security Research Centre (CSCRC) Limited whose activities are partially funded by the Australian Government’s Cooperative Research Centres Programme. We are currently tracking the impact CSCRC funded research. If you have used this code/data in your project, please contact us at contact@cybersecuritycrc.org.au to let us know.

### Citations
Please cite our paper, if use this codebase. Thank you.
