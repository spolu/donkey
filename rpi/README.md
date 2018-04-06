## Install

### Miniconda

https://gist.github.com/simoncos/a7ce35babeaf73f512be24135c0fbafb

wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-armv7l.sh
bash Miniconda3-latest-Linux-armv7l.sh # -> change default directory to /home/pi/miniconda3
vim /home/pi/.bashrc # -> add: export PATH="/home/pi/miniconda3/bin:$PATH"
sudo reboot -h now

### PyTorch

http://book.duckietown.org/master/duckiebook/pytorch_install.html

### Swap

Increase by editing: /etc/dphys-swapfile

### Send code to donkey

rsync -arv --exclude '.*' /Users/stan/src/donkey/rpi pi@d2.local:~/
