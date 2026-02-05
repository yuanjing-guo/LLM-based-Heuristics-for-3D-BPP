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
### 2.1 Single Episode
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
### 2.2Evaluation Mode
Create a folder named 'report' first, then run evaluate.py, this script will automaticlly generate a .txt evaluation report, named like 'eval__timestamp.txt'

Usage:  
Register a heuristic first, then do:  
Run all heuristics for 10 rounds (max_steps=200 per round):
```
python3 evaluate.py --heuristics all --rounds 5 --max_steps 200
```
Run only two heuristics:
```
```  
Save videos for each round (this will be much slower and produce large files):
```
python3 evaluate.py --heuristics all --rounds 5 --save_video
```
### 2.3 buffer and physics-awareness
now the simulation env supports buffer and physics functions. Demo heuristics are 'floor_building_buffer_rule_physics' and 'floor_building_buffer'  
you can explicitly specify the mode by:
```
python3 run.py --heuristic floor_building_buffer_rule_physics --soft
```

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

DEMO
```
uvicorn demo.web.app:app   --reload   --reload-exclude demo/_llm_demo_generated.py   --host 127.0.0.1 --port 8000
```
TEST
```
python run.py --debug --debug_start 0 --seed 0
```
SOFT TEST
```
python run.py --debug --debug_start 0 --seed 0 --soft
```

## 8. Experiment

'''
hand-crafted heuristics:  
first_fit  
best_fit  
corner_point  
extreme_point  
floor_building  
empty_maximal_spaces     #all these heuristics have taken buffer into consideration.

llm_based heuristics:
todo...
## 9. Experimental Setup
Data 1:   
type_dict = {
            1: {"size":(0.03, 0.03, 0.03), "friction":(1.0, 0.005, 0.0001), "density":500, "softness": 0.9, "count": 10, "material":lightwood},
            2: {"size":(0.04, 0.03, 0.03), "friction":(1.0, 0.005, 0.0001), "density":500, "softness": 0.2, "count": 10, "material":lightwood},
            3: {"size":(0.04, 0.04, 0.03), "friction":(1.0, 0.005, 0.0001), "density":5000, "softness": 0.2, "count": 10, "material":darkwood},
            4: {"size":(0.04, 0.04, 0.04), "friction":(1.0, 0.005, 0.0001), "density":5000, "softness": 0.9, "count": 10, "material":darkwood},
        }    

Data 2:   
type_dict = {
            1: {"size":(0.03, 0.03, 0.03), "friction":(1.0, 0.005, 0.0001), "density":500, "softness": 0.9, "count": 10, "material":lightwood},
            2: {"size":(0.035, 0.03, 0.03), "friction":(1.0, 0.005, 0.0001), "density":500, "softness": 0.2, "count": 10, "material":lightwood},
            3: {"size":(0.035, 0.035, 0.03), "friction":(1.0, 0.005, 0.0001), "density":5000, "softness": 0.2, "count": 10, "material":darkwood},
            4: {"size":(0.035, 0.035, 0.035), "friction":(1.0, 0.005, 0.0001), "density":5000, "softness": 0.9, "count": 10, "material":darkwood},
        }

Data 3:   
type_dict = {
            1: {"size":(0.03, 0.03, 0.03), "friction":(1.0, 0.005, 0.0001), "density":500, "softness": 0.9, "count": 10, "material":lightwood},
            2: {"size":(0.025, 0.03, 0.03), "friction":(1.0, 0.005, 0.0001), "density":500, "softness": 0.2, "count": 10, "material":lightwood},
            3: {"size":(0.025, 0.025, 0.03), "friction":(1.0, 0.005, 0.0001), "density":5000, "softness": 0.2, "count": 10, "material":darkwood},
            4: {"size":(0.025, 0.025, 0.025), "friction":(1.0, 0.005, 0.0001), "density":5000, "softness": 0.9, "count": 10, "material":darkwood},
        }