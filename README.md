# object-detection

## Overview

This is a small computer vision project to identify missing & added objects in real time frame. It has two different scripts running two different methods of analyzing the same taks- if something has been removed or added. Lightweight OpenCV based scripts, one with proper GPU acceleration and other works on just CPU.

## Features

1. Can generate a heatmap(outline) of the object that was removed or placed into the frame.
2. Detects humans and ignores them (since they keep moving) so to keep track of only objects and not every indiividual person.
3. GPU-accelerated code that could easily reach high-frame rates & efficient & faster performance.
4. Real-time detection is supported as well, but only non-gpu version works well with it.

## Usage

There are two different scripts/functions that give different outputs, based on your usecase, we'll use them or switch to another

### Detection.py

```python
from detection import run_motion_overlay

run_motion_overlay(
  input_source="data/input.mp4", # 0 for webcam, or provide video file path
  output_path="output.mp4",
  resize_width=640,
  alpha=0.5,
  show_window=True
)
```

### Script.py

```python
from script import run_gpu_motion_overlay

run_gpu_motion_overlay(
  input_path="data/input.mp4",
  output_path="data/overlay.mp4",
  resize_width=640,
  show_window=True
)
```
