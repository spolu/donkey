### Installing

Clone the project repository git clone `keybase://team/dr1ve/donkey`
Install Unity.
Install [mMiniconda](https://conda.io/miniconda.html) to create a python environment. Choose Python 3.6 and follow instructions. Check miniconda is working by running `conda list` after restarting your Terminal Window.
Create a dr1ve python environment `conda create --name dr1ve`
Launch the new python environment `source activate dr1ve`
Install missing modules: `python-socketio`, `eventlet`, `flask`
(When could not find a version that satisfies the requirement `conda install web.py` . I ran `conda install -c conda-forge web.py`)

### Activating Environment

To activate this environment, use `source activate dr1ve`
To deactivate environment, use `source deactivate`
Run `python sim_client.py`
Open Sim folder in Unity