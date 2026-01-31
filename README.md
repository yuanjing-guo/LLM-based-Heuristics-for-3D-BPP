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
python3 evaluate.py --heuristics all --rounds 10 --max_steps 200
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





## 本地模型新增文件说明书

data/lora_sft/train.jsonl
LoRA微调用的数据集目录 git

train_lora.py
LoRA微调主脚本 git

lora_out
LoRA训练产物

merge_lora.py
LoRA合并与模型封装 git

models/
存放本地模型 (很大 但是得用)

Modelfile
Ollama 基础模型定义 git

Modelfile.qwen_lora
Ollama+LoRA模型定义 git

environment_lora.yaml
LoRA环境所需依赖 git


## llm-based heuristic 本地模型使用说明书
## 0前提条件（只要装一次）

Conda 环境：

palletization（跑环境）

Ollama 已安装

模型已存在于 Ollama：   要看到：qwen-heuristic:lora

```
ollama list    
```

## 1启动Ollama

打开一个新终端 而且这个终端要一直开着

```
ollama serve
```

## 2切到正确环境

```
conda activate palletization
cd ~/llm_based_BPP
```

## 3设置环境变量

```
export MUJOCO_GL=glfw

export LLM_API_URL="http://localhost:11434/api/generate"
export LLM_MODEL="qwen-heuristic:lora"
export LLM_API_KEY="dummy"
```

## 4正式启动

```
python run.py --heuristic llm_based --seed 0
```