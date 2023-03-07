# Preventer

Preventing station platform from falling

If you are getting started demo, Please run this command.

### Environment

- windows 10/ 11 Home, Pro
- python 3.9
- Any Core series (10th gen later)

### Setup

Create Python virtual environment

```cmd
python -m venv .venv

.\venv\Scripts\activate
```

### Install Module

The module have already written "requirements.txt"

```cmd
python -m pip install -r requirements.txt
```

### Run example

HumanPoseEstimation 3D(web cam)

```cmd
python HumanPoseEstimation3D.py
```

HumanPoseEstimation 3D(video url)

```cmd
python HumanPoseEstimation.py --source [video path] or [video url] --skip_first_frames [any value]
```
