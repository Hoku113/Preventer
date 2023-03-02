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

comment out `./HumanPoseEstimation3D.py` in `line56` to `line60`

```python
url = "video url" 

video = pafy.new(url, ydl_opts={'nocheckcertificate': True})
best = video.getbest(preftype="mp4")
video_player = VideoPlayer(best.url, flip=False, fps=30, skip_first_frames=0)

video_player.start()
```

and run it

```cmd
python HumanPoseEstimation3D.py
```
