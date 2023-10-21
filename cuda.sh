#!/bin/bash

### steps ####
# verify the system has a cuda-capable gpu
# download and install the nvidia cuda toolkit and cudnn
# setup environmental variables
# verify the installation
###

### to verify your gpu is cuda enable check
lspci | grep -i nvidia

### If you have previous installation remove it first. 
sudo apt-get purge nvidia*
sudo apt remove nvidia-*
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt-get autoremove && sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*

### Make GCC-10 as the default GCC version.
sudo apt update -y
sudo apt upgrade -y
sudo apt-get install libfreeimage3 libfreeimage-dev

# system update
sudo apt-get update -y
sudo apt-get upgrade -y

# install other import packages
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

# first get the PPA repository driver
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update -y

# install nvidia driver with dependencies
sudo apt install nvidia-driver-535 -y

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update -y
sudo apt-get -y install cuda

#wget https://developer.nvidia.com/compute/cudnn/secure/8.6.0/local_installers/11.8/cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update -y
sudo apt-get install libcudnn8=8.6.0.163-1+cuda11.8 -y
sudo apt-get install libcudnn8-dev=8.6.0.163-1+cuda11.8 -y
sudo apt-get install libcudnn8-samples=8.6.0.163-1+cuda11.8 -y

sudo apt update -y
sudo apt upgrade -y

sudo nano $HOME/.profile
sudo nano $HOME/.bashrc

# export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# nvidia-smi
# nvcc -V

# pip uninstall MarkupSafe absl-py astunparse cachetools flatbuffers gast google-auth google-auth-oauthlib google-pasta grpcio h5py jax keras libclang markdown ml-dtypes numpy opt-einsum packaging protobuf pyasn1 pyasn1-modules requests-oauthlib rsa scipy tensorboard tensorboard-data-server tensorflow tensorflow-estimator tensorflow-io-gcs-filesystem termcolor typing-extensions werkzeug wrapt
