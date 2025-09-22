# Preventer
Preventing station platform from falling

## Version
Current Version: **0.1.0-beta**

Released: **2025-09-22**

## Environment
- windows 10/11 Home/Pro
- python 3.9+
- Intel Core series (7th later)

## Setup
Create Python virtual environment

```cmd
python -m venv .venv

.\venv\Scripts\activate
```

## Install Module
The module have already written "requirements.txt"

```cmd
python -m pip install -r requirements.txt
```

## Run example
HumanPoseEstimation 3D(web cam)

```cmd
python HumanPoseEstimation3D.py
```

HumanPoseEstimation 3D(video url)

```cmd
python HumanPoseEstimation.py --source [video path] or [video url] --skip_first_frames [any value]
```
