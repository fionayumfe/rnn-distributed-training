

### Distributed Training of RNN by Spark

This prototype is trying to speed up model training for large recurrent neural network. Imagine you want to train your RNN for a large vocabulary,say tens of thousands of frequent words, you don't want to wait for
a couple of hours for training. Also you aim to find global optimal model parameters. Why not train your models by workers distributed on a Hadoop cluster? By that way, you will have the chance to pick up models trained in parallel and significantly speed up your training. This prototype is built on top of Spark 2.2. It can be run on AWS EMR clusters to test performance. For prototype purpose, I want to dive in details of code to figure out which is the bottleneck. That is why I didn't use either TensorFlow or Theano packages. Instead, I built the prototype on top of this excellent <a href= "http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/">tutorial.</a>  


### Standalone Installation Guide:

```bash
# Clone the repo
git clone https://github.com/dennybritz/rnn-tutorial-rnnlm
cd rnn-tutorial-rnnlm

# Create a new virtual environment (optional, but recommended)
virtualenv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

```

### Setting up a CUDA-enabled GPU instance on AWS EC2:

```bash
# Install build tools
sudo apt-get update
sudo apt-get install -y build-essential git python-pip libfreetype6-dev libxft-dev libncurses-dev libopenblas-dev  gfortran python-matplotlib libblas-dev liblapack-dev libatlas-base-dev python-dev python-pydot linux-headers-generic linux-image-extra-virtual
sudo pip install -U pip

# Install CUDA 7
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1410/x86_64/cuda-repo-ubuntu1410_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1410_7.0-28_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda
sudo reboot

# Clone the repo and install requirements
git clone git@github.com:dennybritz/nn-theano.git
cd nn-theano
sudo pip install -r requirements.txt

# Set Environment variables
export CUDA_ROOT=/usr/local/cuda-7.0
export PATH=$PATH:$CUDA_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
# For profiling only
export CUDA_LAUNCH_BLOCKING=1

```
