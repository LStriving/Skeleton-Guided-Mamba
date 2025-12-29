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
# set cuda/path/LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
```

```bash
conda create -n sg-mamba python=3.10
```

### Install PyTorch

```bash
conda activate sg-mamba
conda install gxx_linux-64
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CC=$(which x86_64-conda-linux-gnu-gcc)
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia 
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# CUDA 12.1
# conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 11.8
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

```

### Install Mamba

Note that these steps need to be reinstalled if you change Torch or CUDA versions.
```bash


cd causal-conv1d
pip install . # an efficient implementation of a simple causal Conv1d layer used inside the Mamba block.
# pip install --no-build-isolation . # try this when failed.
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
conda install ffmpeg -c conda-forge -y
pip install opencv-python scikit-video numpy==1.23.5 opencv-contrib-python matplotlib pandas fairscale timm calflops pykalman joblib h5py mmengine tensorboard
```


```bash
# set the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```