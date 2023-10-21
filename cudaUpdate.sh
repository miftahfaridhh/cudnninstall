#!/bin/bash

### steps ####
# verify the system has a cuda-capable gpu
# download and install the nvidia cuda toolkit and cudnn
# setup environmental variables
# verify the installation
###

pip uninstall MarkupSafe absl-py astunparse cachetools flatbuffers gast google-auth google-auth-oauthlib google-pasta grpcio h5py jax keras libclang markdown ml-dtypes numpy opt-einsum packaging protobuf pyasn1 pyasn1-modules requests-oauthlib rsa scipy tensorboard tensorboard-data-server tensorflow tensorflow-estimator tensorflow-io-gcs-filesystem termcolor typing-extensions werkzeug wrapt -y

### to verify your gpu is cuda enable check
lspci | grep -i nvidia

### If you have previous installation remove it first. 
sudo apt-get purge nvidia* -y
sudo apt remove nvidia-* -y
sudo rm /etc/apt/sources.list.d/cuda* -y
sudo rm -rf /usr/lib/cuda* -y
sudo rm -rf /usr/include/cuda -y
sudo apt-get autoremove -y && sudo apt-get autoclean -y

### Make GCC-10 as the default GCC version.
sudo apt update -y
sudo apt upgrade -y
sudo apt-get install libfreeimage3 libfreeimage-dev -y

# system update
sudo apt-get update -y
sudo apt-get upgrade -y

# install other import packages
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev -y

# first get the PPA repository driver
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update -y

# install nvidia driver with dependencies
sudo apt install nvidia-driver-535 -y


#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
#sudo dpkg -i cuda-keyring_1.0-1_all.deb
#sudo apt-get update
#sudo apt-get -y install cuda


wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda


#wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
#sudo sh cuda_11.8.0_520.61.05_linux.run


sudo apt update -y
sudo apt upgrade -y


sudo cp cudnn/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cudnn/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/include:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> $HOME/.profile
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> $HOME/.profile
source $HOME/.profile

sudo apt update -y
sudo apt upgrade -y

pip install tensorflow-2.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

sudo apt update -y
sudo apt upgrade -y

# nvidia-smi
nvcc -V
