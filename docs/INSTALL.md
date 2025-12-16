# Installation

## Requirements

- Linux
- NVIDIA GPU
- PyTorch 1.12+
- CUDA 11.6+

## Clone the repository

```bash
git clone https://github.com/LStriving/Skeleton-Guided-Mamba.git --recursive
cd Skeleton-Guided-Mamba
```

## Install dependencies

### Create a virtual environment (Conda)

```bash
conda create -n sg-mamba python=3.10
```

### Install PyTorch

```bash
conda activate sg-mamba
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
# conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Install Mamba

```bash
cd causal-conv1d
pip install . # an efficient implementation of a simple causal Conv1d layer used inside the Mamba block.
cd ../mamba
pip install -e . # the core Mamba package. (you should not run `pip install mamba-ssm` here since it is different from the original mamba-ssm package)
```

### Install other dependencies 

#### NMS
```bash
cd ../sg-mamba/libs/utils
python setup.py install --user
```

### Install remaining Python packages

```bash
pip install opencv-python scikit-video numpy==1.23.5 opencv-contrib-python matplotlib
```