[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
# Hilbert Diffusion Model

This is an official repository for `Hilbert Diffusion Model (HDM)` based on the paper *Score-based Generative Modeling through Stochastic Evolution Equations in Hilbert Spaces*, **Lim et al.**, 2023.

## Important Configurations (.yml)
```
data:
    modality: "2D" # 1D or 2D
    dataset: "CIFAR10" # Quadratic, Melbourne, Gridwatch, MNIST, CIFAR10, LSUN, AFHQ, FFHQ

training:
    ckpt_store: 5000

sampling:
    batch_size: 64
```

## Dataset setting 
AFHQ
```
bash download.sh afhq-dataset
```
LSUN-church
```
python download_lsun.py --category church_outdoor
```
FFHQ

Download FFHQ dataset through "https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py"

## Code Usage
### Dependencies
Run the following command to install the necessary Python packages for our code. Note that we tested our model in Python 3.8.16.
```
pip install -r requirements.txt
```


**Single GPU training command example**
```
python main.py --config quadratic_fno.yml --exp ./outs/quadratic_1d_exp
```

**Multiple GPU training command example**
```
torchrun --nproc_per_node=NUMBER_OF_GPUS main.py --config quadratic_fno.yml --exp quadratic_1d_exp --distributed
```

`main.py` requires the following arguments for training a model.

```
main.py 
--config: Configuration file stored in "./configs"
--exp: The folder name to store checkpoints, tensorboard logger. (recommend to use "./outs/MODALITY_DATASET")
--distributed: whether to train a model using multiple GPUs. (default is set to False)
--resume: Whether to resume training from the last checkpoint. (No resume in the 1D dataset)
```

### Sampling
Using trained checkpoints, the sampling command requires the following arguments.

```
main.py 
--config: Configuration file stored in "./configs"
--exp: The folder name to store checkpoints, tensorboard logger. (recommend to use "./outs/MODALITY_DATASET")
--distributed: whether to train a model using multiple GPUs. (default is set to False)
--nfe: Number of function evaluations. (default is set to be 1,000)
--fid: Whether to calculate FID scores. (default is set to be False)
--prior: whether to use HDM prior or IHDM prior. (default is set to be 'hdm')
--sample_type: Whether to comduct Imputation or Super-resolution sampling for 2D image experiments. (default is set to be 'sde')
--degraded_type: Whether to use Gaussian blur or pixelate to generate degraded images for Super-resolution task. (default is set to be 'blur')
```

### Multi-GPU
Our code supports both single and multi-GPU training codes. We recommended using TorchElastic to run multi-GPU training. For example, using a single node with multiple GPUS, you can try the following command. 
```
torchrun --standalone --nnodes=NUMBER_OF_NODE --nproc_per_node=NUMBER_OF_GPUS main.py (--arg1, ..., -args) --distributed
```

### Using ckpt for sampling
[**HDM Checkpoints Google Drive Link**](https://drive.google.com/drive/folders/1dsrqHasAvNulLwJ8dxWjj3B6t-i4ouUA?usp=sharing)

ckpt_dir = "path" 

Quadatic
```
CUDA_VISIBLE_DEVICES=0  LOCAL_RANK=0 torchrun --nproc_per_node=1 --master_port=12422 main.py --config quadratic_fno.yml --sample  
```
AFHQ
```
CUDA_VISIBLE_DEVICES=0  LOCAL_RANK=0 torchrun --nproc_per_node=1 --master_port=12422 main.py --config afhq_ddpm.yml --sample --distributed
```
FFHQ
```
CUDA_VISIBLE_DEVICES=0  LOCAL_RANK=0 torchrun --nproc_per_node=1 --master_port=12422 main.py --config ffhq_ddpm.yml --sample --distributed
```
LSUN
```
CUDA_VISIBLE_DEVICES=0  LOCAL_RANK=0 torchrun --nproc_per_node=1 --master_port=12422 main.py --config lsun_ddpm.yml --sample --distributed
```
## Empirical Results

- 1D curve generation (Quadratic and Melbourne Dataset)

<img src='exp_fig_results/quadratic.png'/>

<img src='exp_fig_results/melbourne.png'/>

<img src='exp_fig_results/gridwatch.png'/>

- Resolution-free image generation

- Motion generation

## Citation
If you find this code useful for your research, please cite our paper:
```
TBA
```
