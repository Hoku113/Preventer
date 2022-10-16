# Preventer

Preventing station platform from falling

If you are getting started demo, Please run this command.

### Environment

- windows 10/ 11 Home, Pro
- python 3.9

### Setup

Create Python virtual environment

```python
python -m venv .venv

.\venv\Scripts\activate
```

### Install Module

The module have already written "requirements.txt"

```python
python -m pip install -r requirements.txt
```

### Install model

Using `omz_downloader` command, you can downloaded another AI model.

```cmd
omz_downloader --name <model_name> --precision <precision> --output_dir <output_model_path>
```

※1 precision example value: `FP16`, `INT8`, `FP16-INT8`
※2 By default, some models are already downloaded
`human-pose-estimation-0001`, `action-recognition-0001`

### Run example

HumanPoseEstimation

```python
python HumanPoseEstimation.py
```

ActionRecognition.py

```python
python ActionRecognition.py
```