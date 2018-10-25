
### Runbook Competition

# Make a capture to check Canny parameters

From `donkey/` directory:

```
mkdir ../captures/capture_20181012_1514
OMP_NUM_THREADS=2 python3 raspi_runner.py ./configs/raspi-reinforce_20180927_1200.json --reinforce_load_dir=/home/pi/models/reinforce_20180927_1200/ --driver_optical_flow_speed=16 --driver_fixed_throttle=0.50 --capture_dir=../captures/capture_20181012_1514 
```

From `donkeuy/` directory:
```
python3 capture_viewer.py --capture_set_dir=../captures/ 
```

Connect from your browser to:
```
http://dr1ve.local:9092/static/viewer.html?capture=capture_20181012_1514
```

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

wget https://beta.unity3d.com/download/170f0691b973/UnitySetup-2018.2.7f1
chmod +x UnitySetup-2018.2.7f1
mkdir -p ~/opt/unity3d
./UnitySetup-2018.2.7f1 --unattended --install-location=~/opt/unity3d

```

Set the following env variables in `.bashrc`:
```
UNITY_SERIAL=
UNITY_USERNAME=
UNITY_PASSWORD=
```

### At reboot

```
sudo bash
./scripts/trainer_setup_root.sh
```

## Raspberry

Connect to raspi
'ssh pi@dr1ve.local' or 'sssh pi@<your_pi_ip_address>'

Push repo to donkey
`rsync -arv --exclude '.*' --exclude 'build' --exclude 'sim'  ~/code/donkey   pi@dr1ve.local:~/`

Safe
```
OMP_NUM_THREADS=2 python3 raspi_runner.py ./configs/raspi-reinforce_20180927_1200.json --reinforce_load_dir=/home/pi/models/reinforce_20180927_1200/ --driver_optical_flow_speed=18 --driver_fixed_throttle=0.55```

Agressive
```
OMP_NUM_THREADS=2 python3 raspi_runner.py ./configs/raspi-reinforce_20181001_1700.json --reinforce_load_dir=/home/pi/models/reinforce_20181001_1700/ --driver_optical_flow_speed=14 --driver_fixed_throttle=0.80```
