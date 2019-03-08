#!/usr/bin/env bash

#### Virtual Env ####
conda create -p gomut python=3.6
conda install -p gomut pip -c anaconda
#conda install -p gomut pytorch torchvision cuda80 -c pytorch
conda install -p gomut pytorch-cpu torchvision-cpu -c pytorch

source activate gomut/

pip install --upgrade pip

pip install numpy
pip install pandas
pip install scipy
pip install tensorflow
pip install tensorboard
pip install tensoarboardX
pip install prody
pip install biopython
pip install h5py
pip install opencv-python
pip install pillow
pip install ipython==6.2.0 notebook

conda install -p gomut -c conda-forge matplotlib
conda install -p gomut -c conda-forge tqdm


### Download and compile STRIDE
# http://webclu.bio.wzw.tum.de/stride/install.html
