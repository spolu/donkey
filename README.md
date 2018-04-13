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

- To activate this environment, use `source activate dr1ven`
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

- Run `python runner.py configs/stan_ppo.json --simulation_headless=false
  --load_dir=/keybase/team/dr1ve/experiments/exp_20180321_0721/`.

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

wget http://beta.unity3d.com/download/3c89f8d277f5/UnitySetup-2017.3.0f1
chmod +x UnitySetup-2017.3.0f1
mkdir -p ~/opt/unity3d
./UnitySetup-2017.3.0f1 --unattended --install-location=~/opt/unity3d

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
# For K80
sudo nvidia-smi -ac 2505,875
sudo nvidia-smi --auto-boost-default=DISABLED

sudo bash
/usr/bin/X :1 &
```

### Useful bash scripts

```
push-donkey () {
  rsync -arv --exclude '.*' ~/src/donkey spolu-dev@$1:~/src/
}
pull-experiment() {
  rsync -arv spolu-dev@$1:/tmp/$2 /keybase/team/dr1ve/experiments/
}
push-seed () {
  EXPERIMENT=`date +'%Y%m%d_%H%M'`
  rsync -arv /keybase/team/dr1ve/experiments/$1 spolu-dev@$2:/tmp/exp_$EXPERIMENT
}
```


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

## Read list

https://www.youtube.com/watch?v=b_lBL2yhU5A&feature=youtu.be
