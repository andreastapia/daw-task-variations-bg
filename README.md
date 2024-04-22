# Simulation of Two-Step tasks using a three-loop hierarchical model of the basal ganglia

Code for thesis "Simulación de experimentos de selección de acciones con tareas de dos etapas utilizando un modelo computacional jerárquico de los ganglios basales".

Based on the work of Baladron & Hamker (2020) and Schaal (2022) masters thesis.

The proyect have simulations for the Daw Task (2011), Dezfouli Task (2013) & Doll Task (2015) using ANNarchy framework.

Below is a description of the folders:

- basal_ganglia: main folder of simulations, results & visualization scripts.
    - model: contains the full model for Daw, Dezfouli & Doll tasks.
    - optuna: several tries to find a set of parameter for the model using optuna. They are executed with the run_optuna.sh file. For these simulations you need to configure a MySQL database somewhere.
    - simulations: contains all the simulations of the tasks. Simulations for individual loops contains the model inside the same file.
    - testing: contains testing scripts, some of them were used, some of the were discarded.
    - utils: contains scripts for generating the visualization of the data depending on the results*.
- schaal_og_script: contains the original code of Schaal thesis.

*Simulation results will be stored on basal_ganglia/results

**All simulations can be executed using the main.py file inside the basal_ganglia folder.**

## Previous installations

```bash
sudo apt install build-essential gcc git python3-dev python3-setuptools python3-pip python3-numpy python3-scipy python3-matplotlib cython3

sudo apt install python3-pyqtgraph python3-pyqt5.qtopengl python3-lxml pandoc 
```

## Create virtual env and activate

```bash

pip3 install virtualenv

python3 -m venv venv && source venv/bin/activate

```

## install dependencies

```bash

pip install --upgrade pip setuptools numpy sympy tensorboardx scipy

pip install -r requirements.txt

pip install ANNarchy

```

## run experiments

Modify file run.sh, change to the experiment declared in basal_ganglia/main.py, modify the amount of simulations and run. For parallel executions just open multiple terminals, you can automatize the task too.

```bash
chmod +x run.sh 
```

```bash
./run.sh 
```