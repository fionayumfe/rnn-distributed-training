

### Distributed Training of RNN by Spark

This prototype is trying to speed up model training for large recurrent neural network. Imagine you want to train your RNN for a large vocabulary,say tens of thousands of frequent words, you don't want to wait for
a couple of hours for training. Also you aim to find global optimal model parameters. Why not train your models by workers distributed on a Hadoop cluster? By that way, you will have the chance to pick up models trained in parallel and significantly speed up your training. This prototype is built on top of Spark 2.2. It can be run on AWS EMR clusters to test performance. For prototype purpose, I want to dive in details of code to figure out which is the bottleneck. That is why I didn't use either TensorFlow or Theano packages. Instead, I built the prototype on top of this excellent <a href= "http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/">tutorial.</a>  


### Standalone Installation Guide:

```bash
# Clone the repo
git clone https://github.com/fionayumfe/rnn-distributed-training.git
cd rnn-tutorial-rnnlm

# Create a new virtual environment (optional, but recommended)
virtualenv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

```
### Setting up a Elastic Map-Reduce (EMR) cluster on AWS:
```
aws emr create-cluster --configurations your-json-file --release-label emr-5.3.1 --instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m3.xlarge InstanceGroupType=CORE,InstanceCount=1,InstanceType=m3.xlarge --auto-terminate
```
You can also add spark-submit step to your emr script
```
$SPARK_HOME/bin/spark-submit \
    --master   yarn
    --conf     spark.yarn.submit.waitAppCompletion=false
    --conf     spark.executorEnv.PYTHONHASHSEED=0
    --conf     spark.yarn.executor.memoryOverhead=4096
    --conf     spark.executor.memory=7.5g
    --packages org.apache.hadoop:hadoop-aws:2.7.3
/home/hadoop/spark_main.py
```
You may also create a shell script as bootstrap step. In the step, all your source files will be copied to your data node and your dependencies will be installed as well.
An example bootstrap file is

```bash
#!/usr/bin/env bash

aws s3 cp   s3://your_folder/spark_main.py             /home/hadoop/

export PATH="$PATH:/home/hadoop"
export CLASS_PATH="$CLASS_PATH:/home/hadoop"
export PYTHONHASHSEED=0
alias  python=python34

sudo yum -y install your packages
#install dependencies (Non-standard and non-Amazon Machine Image Python modules)
sudo pip-3.4 install py4j boto3  psutil awscli pandas

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
git clone https://github.com/fionayumfe/rnn-distributed-training.git
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
