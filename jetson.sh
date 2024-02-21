####CUDA 11.4
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/sbsa/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda-repo-ubuntu1804-11-4-local_11.4.4-470.82.01-1_arm64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-4-local_11.4.4-470.82.01-1_arm64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-11-4-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
sudo apt-get -o Dpkg::Options::="--force-overwrite" install --fix-broken -y
dpkg -l | grep cuda


####cudNN 8.2.4
#download from source https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.4/11.4_20210831/cudnn-11.4-linux-aarch64sbsa-v8.2.4.15.tgz

tar -xzvf cudnn-11.4-linux-aarch64sbsa-v8.2.4.15.tgz
mv cuda cudnn
sudo cp cudnn/include/cudnn*.h /usr/local/cuda-11.4/include 
sudo cp -P cudnn/lib64/libcudnn* /usr/local/cuda-11.4/lib64 
sudo chmod a+r /usr/local/cuda-11.4/include/cudnn*.h /usr/local/cuda-11.4/lib64/libcudnn*

echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.4/include:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

echo 'export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}' >> $HOME/.profile
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> $HOME/.profile
source $HOME/.profile

sudo apt update && sudo apt upgrade -y

nvcc -V


##python 3.10

sudo apt-get install curl gcc libbz2-dev libev-dev libffi-dev libgdbm-dev liblzma-dev libncurses-dev libreadline-dev libsqlite3-dev libssl-dev make tk-dev wget zlib1g-dev gdebi-core -y

export PYTHON_VERSION=3.10.13
export PYTHON_MAJOR=3

curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
tar -xvzf Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION}

./configure --prefix=/opt/python/${PYTHON_VERSION} --enable-shared --enable-optimizations --enable-ipv6 LDFLAGS=-Wl,-rpath=/opt/python/${PYTHON_VERSION}/lib,--disable-new-dtags
make
sudo make install

#curl -O https://bootstrap.pypa.io/get-pip.py
#/opt/python/${PYTHON_VERSION}/bin/python${PYTHON_MAJOR} get-pip.py

/opt/python/${PYTHON_VERSION}/bin/python${PYTHON_MAJOR} --version

echo 'export PATH=/opt/python/3.10.13/bin/:$PATH' >> $HOME/.profile
source $HOME/.profile

python3.10 -V

python3.10 -m pip install --upgrade pip

python3.10 -m pip install -r ~/edge-detection-occ/onnxyoloobjdetect/requirement.txt

python3.10 -c "import onnxruntime as ort; print(ort.get_device())"


###install onnxruntime-gpu
python3.10 -m pip uninstall onnxruntime
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
git checkout v1.14.1
cd ~
sudo apt install -y --no-install-recommends build-essential software-properties-common libopenblas-dev libpython3.6-dev python3-pip python3-dev python3-setuptools python3-wheel
pip3 install packaging
##cmake update
sudo apt-get remove --purge cmake -y
wget https://github.com/Kitware/CMake/releases/download/v3.29.0-rc1/cmake-3.29.0-rc1.tar.gz
tar -xvf cmake-3.29.0-rc1.tar.gz
cd cmake-3.29.0-rc1
./bootstrap
make -j4
sudo make install
cd ~/onnxruntime
./build.sh --config Release --update --build --parallel --build_wheel --use_cuda --cuda_home /usr/local/cuda-11.4 --cudnn_home /usr/local/cuda-11.4/lib64 
cd ./build/Linux/Release/dist
python3.10 -m pip install onnxruntime_gpu-1.14.1-cp38-cp38-linux_aarch64.whl

