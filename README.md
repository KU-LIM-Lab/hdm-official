
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
# Hilbert Diffusion Model (HDM)

This is an official repository for `Hilbert Diffusion Model (HDM)` based on the paper *Score-based Generative Modeling through Stochastic Evolution Equations in Hilbert Spaces*, Lim et al., ***NeurIPS (Spotlight)***, 2023.

## Code Usage

### Dependencies
Run the following command to install the necessary packages. 
```
# Python 3.8.16
pip install -r requirements.txt
```

### Training

**Configuration**
- `dataset`: cifar10, afhq, ffhq, lsun, quadratic, melbourne, gridwatch
- `architecture`: unet, fno

**Single GPU training**
Run the following code with `hdm_{dataset}_{architecture}.yml`.
```
python main.py --config {config_name}.yml --exp {folder_name}
```

**Multiple GPU training**
Run the following code with `hdm_{dataset}_{architecture}.yml`.
```
torchrun --nproc_per_node=NUMBER_OF_GPUS main.py --config {config_name}.yml --exp {folder_name} --distributed
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
Our code supports both single and multi-GPU training codes. We recommended using `TorchElastic` to run multi-GPU training. For example, using a single node with multiple GPUs, you can try the following command. 
```
torchrun --standalone --nnodes=NUMBER_OF_NODE --nproc_per_node=NUMBER_OF_GPUS main.py {--args} --distributed
```

## Citation
If you use this code in your research, please cite our [paper](https://openreview.net/forum?id=GrElRvXnEj).
```
@inproceedings{
lim2023scorebased,
title={Score-based Generative Modeling through Stochastic Evolution Equations},
author={Lim, Sungbin and Yoon, Eunbi and Byun, Taehyun and Kang, Taewon and Kim, Seungwoo and Lee, Kyungjae and Choi, Sungjoon},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=GrElRvXnEj}
}
```

## Contact
- Sungbin Lim, [sungbin@korea.ac.kr](sungbin@korea.ac.kr)