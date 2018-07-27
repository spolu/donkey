## Getting Started 

- Clone the project repository git clone `keybase://team/dr1ve/donkey`.
- Install [mMiniconda](https://conda.io/miniconda.html) to create a python
  environment. Choose Python 3.6 and follow instructions. Check miniconda is
  working by running `conda list` after restarting your Terminal Window.
  Create a dr1ve python environment `conda create --name dr1ven`.
- Launch the new python environment `source activate dr1ven`.
- Install Unity.
- Install modules: `python-socketio`, `eventlet`, `flask`.

### Python Environment 

- To activate this environment, use `source activate dr1ve`
- To deactivate environment, use `source deactivate`

### Building simulation

- Run `make clean && make simulation` in terminal to build the donkey
  simulator.

### Running human client

- Run `python human.py`.
- Connect with your browser to `http://127.0.0.1:9091/static/index.html`.

### Training a model

- Run `python trainer.py configs/stan_ppo.json`

### Playing a trained model

- Run `python runner.py configs/stan_ppo.json --simulation_headless=false --load_dir=/keybase/team/dr1ve/experiments/exp_20180426_1257/`.

### Editing Simulator in Unity 

- Open `sim/` folder in Unity.
- Run `make clean && make simulation' in terminal to build the donkey
  simulator.

## Setup Google Cloud Compute

```
sudo add-apt-repository ppa:graphics-drivers
sudo apt-get update
sudo apt-get install nvidia-387

wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1704/x86_64/cuda-repo-ubuntu1704_9.1.85-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1704_9.1.85-1_amd64.deb                            
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1704/x86_64/7fa2af80.pub
sudo apt-get update                                                             
sudo apt-get install cuda 

[restart]

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
conda create --name dr1ven
source activate dr1ven
conda install pytorch torchvision cuda91 -c pytorch
pip install eventlet python-socketio flask opencv-python

sudo apt-get update
sudo apt-get install libgtk2.0-0 libnss3 xserver-xorg libgconf-2-4 gnuplot htop libarchive13 make

wget https://beta.unity3d.com/download/170f0691b973/UnitySetup-2018.1.0f2
chmod +x UnitySetup-2018.1.0f2
mkdir -p ~/opt/unity3d
./UnitySetup-2018.1.0f2 --unattended --install-location=~/opt/unity3d

```

Set the following env variables in `.bashrc`:
```
UNITY_SERIAL=
UNITY_USERNAME=
UNITY_PASSWORD=
```

### At reboot

```
sudo nvidia-smi -pm 1

sudo bash
/usr/bin/X :1 &
```

### Useful bash scripts

```
push-donkey () {
  rsync -arv --exclude '.*' ~/src/donkey spolu-dev@$1:~/src/
}
pull-donkey() {
  rsync -arv spolu-dev@$1:/tmp/$2 /keybase/team/dr1ve/experiments/
}
push-seed () {
  EXPERIMENT=`date +'%Y%m%d_%H%M'`
  rsync -arv /keybase/team/dr1ve/experiments/$1 spolu-dev@$2:/tmp/exp_$EXPERIMENT
}
```

## o7

python capture_simulation.py --capture_dir=/tmp/capture_train_20180515/
http://localhost:9091/static/index.html

python capture_trainer.py --capture_dir=/tmp/capture_train_20180515 --save_dir=/tmp/capture_model_20180515/

python runner_simulation.py --load_dir=/tmp/capture_model_20180515/

### Resources

https://cloud.google.com/compute/docs/gpus/add-gpus#install-driver-script
https://towardsdatascience.com/how-to-run-unity-on-amazon-cloud-or-without-monitor-3c10ce022639

## Raspberry Pi 3

### Miniconda

https://gist.github.com/simoncos/a7ce35babeaf73f512be24135c0fbafb

wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-armv7l.sh
bash Miniconda3-latest-Linux-armv7l.sh # -> change default directory to /home/pi/miniconda3
vim /home/pi/.bashrc # -> add: export PATH="/home/pi/miniconda3/bin:$PATH"
sudo reboot -h now

### Swap

Increase by editing: /etc/dphys-swapfile
If not enough increase by editing /sbin/dphys-swapfile

### PyTorch

Install numpy first

http://book.duckietown.org/master/duckiebook/pytorch_install.html

```
export NO_DISTRIBUTED=1
export NO_CUDA=1
MAX_JOBS=1 python setup.py build
sudo -E python setup.py install
```

### Send code to donkey

rsync -arv --exclude '.*' --exclude 'build' --exclude 'sim' ~/src/donkey pi@dr1ve.local:~/

### Run

Human driver with web client
```
python raspi.py configs/raspi.json
```

Model
```
python raspi.py configs/raspi.json --load_dir=/home/pi/exp_20180407_1537/
```

### Save image of donkey

`diskutil list`
`dd if=/dev/disk2 of=~/SDCardBackup.dmg`
https://thepihut.com/blogs/raspberry-pi-tutorials/17789160-backing-up-and-restoring-your-raspberry-pis-sd-card

## Jetson TX2

### Flash Jetpack

Follow guidance to flash JetPack to NVIDIA TX2 with Mac through a VirtualBox (VM)
https://github.com/KleinYuan/tx2-flash
Doesn't work through wifi, jetson needs to be conencted on ethernet from my perspective

### Create virtual env
`sudo apt-get update && sudo apt-get upgrade`
`sudo apt-get install -y python-pip`
`pip install virtualenvwrapper`
`sudo apt-get install virtualenv`

`cat >> ~/.bash_profile
export WORKON_HOME=~/Envs
source ./.local/bin/virtualenvwrapper.sh`

`mkdir -p $WORKON_HOME`
`mkvirtualenv -p python3 dr1ve`

### Install PyTorch

https://github.com/pytorch/pytorch#binaries
Get the source
`git clone --recursive https://github.com/pytorch/pytorch`
and install
`python setup.py install`

note:  pyTorch documentation calls for use of Anaconda, however Anaconda isn't available for aarch64

## Read list

https://www.youtube.com/watch?v=b_lBL2yhU5A&feature=youtu.be
