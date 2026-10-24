# ascii-cam

Windows webcam filter that renders you as ASCII / Matrix-style digital rain and
publishes it as a virtual camera (or a preview window you can screen-share).

## Setup

1. Install [OBS Studio](https://obsproject.com) once (provides the virtual camera driver).
2. Create the environment:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Run

```
.venv\Scripts\python ascii_cam.py --rain --face --solo --font-size 10 --out-size 1920x1080 --invert-amount 0 --preview --fullscreen
```

Pick **OBS Virtual Camera** in Zoom / Meet, or screen-share the **ASCII Cam** window
for a sharper picture (video calls compress small text heavily).

Preview keys: `f` toggles fullscreen, `q` / `Esc` quits.

## Options

| Flag | Meaning |
| --- | --- |
| `--rain` | Matrix digital rain with your image in it |
| `--matrix` | Still green-on-black ASCII portrait |
| `--color` | Tint characters with the source colour |
| `--face` | Crop to your face (`--zoom` sets how tight) |
| `--solo` | Remove the background |
| `--font-size N` | Smaller = more detail |
| `--out-size WxH` | Output size, e.g. `1920x1080` |
| `--gain`, `--smooth`, `--speed`, `--invert-amount` | Brightness, steadiness, rain speed, invert |
| `--preview`, `--fullscreen` | Show a window |

Notes: OpenCV is pinned to 4.x (5.0 dropped the face detector) and mediapipe to
0.10.21 (newer releases removed the bundled segmentation model).
