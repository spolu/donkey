### Installing 

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
