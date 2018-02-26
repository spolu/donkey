# Setup Google Cloud Compute

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

sudo nvidia-smi -pm 1

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
conda create --name dr1ven
source activate dr1ven
conda install pytorch torchvision cuda91 -c pytorch
pip install eventlet python-socketio flask opencv-python

wget http://beta.unity3d.com/download/3c89f8d277f5/UnitySetup-2017.3.0f1
chmod +x UnitySetup-2017.3.0f1
mkdir -p ~/opt/unity3d
./UnitySetup-2017.3.0f1 --unattended --install-location=~/opt/unity3d
sudo apt-get install libnss3 xserver-xorg libgconf-2-4
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
```

Set the following env variables in `.bashrc`:
```
UNITY_USERNAME=
UNITY_PASSWORD=
UNITY_SERIAL=
```

At startup, start a new X server on :1
```
sudo /usr/bin/X :1 &
```

# Useful resources

https://cloud.google.com/compute/docs/gpus/add-gpus#install-driver-script
https://towardsdatascience.com/how-to-run-unity-on-amazon-cloud-or-without-monitor-3c10ce022639
