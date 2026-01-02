## 1. Environment Setup

### 1.1 Create Conda Environment

```
conda env create -f environment.yaml
conda activate palletization
```

### 1.2 Test

Run:
```
python run.py --heuristic floor_building
```
If the simulation starts and a video is generated under video/, the environment setup is correct.

## 2. Quick Start
Run a single episode with a selected heuristic:

```
python run.py --heuristic floor_building
```

During execution:  
The heuristic generates an action at each step  
The environment executes the action exactly  
Physics simulation determines stability  
Step-wise logs are printed to the terminal  
A rollout video is saved automatically

## 3. Project Structure
```
.
├── assets
├── environment.yaml
├── env.py
├── evaluate.py
├── helpers
│   ├── box_init_pose.npy
│   ├── controller.json
│   ├── material.py
│   └── task_config.py
├── heuristics
│   ├── base.py
│   ├── feasibility.py
│   ├── floor_building.py
│   ├── __init__.py
│   ├── largest_volume_lowest_z.py
│   ├── llm_based.py
│   └── __pycache__
│       ├── base.cpython-38.pyc
│       ├── feasibility.cpython-38.pyc
│       ├── floor_building.cpython-38.pyc
│       ├── handcrafted.cpython-38.pyc
│       ├── largest_volume_lowest_z.cpython-38.pyc
│       └── random_baseline.cpython-38.pyc
├── README.md
├── results
├── run.py
├── slides
│   └── Team_Report1.pptx
└── video
```

## 4. File and Folder Description
env.py:  
Core palletization environment.  
Executes actions exactly as provided by heuristics

run.py:  
Main entry point for running experiments.  
Most users only need to run this file.

heuristics/  
All heuristic planners are implemented here.    

To add a new heuristic:  
Create a new file in this folder  
Implement the heuristic class  
Register it in run.py

helpers/  
Task and simulation configuration files.

video/  
Saved rollout videos. Ignored by git (not committed)

assets/  
Psuedo codes and papers related.

## 5. Development Notes
Do not commit:  
video/  
logs/  
large binary files  
New heuristics should be added under heuristics/  
No changes to env.py are required when adding heuristics  
All heuristics must follow the same action interface

## 6. Common Issues

EGL / OpenGL errors  
Ensure GPU drivers are installed  
Use offscreen rendering

## 7. Notes
This repository is intended as a shared experimental platform.  
Please keep changes modular and avoid breaking existing heuristics.  
Built and tested on Ubuntu 22.04
