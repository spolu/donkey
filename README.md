
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

