from detection import run_motion_overlay

run_motion_overlay(
  input_source=0, # 0 for webcam, or provide video file path
  output_path="output.mp4",
  resize_width=640,
  alpha=0.5,
  show_window=True,
  real_time=False
)

# from script import run_gpu_motion_overlay

# run_gpu_motion_overlay(
#   input_path="data/input.mp4",
#   output_path="data/overlay.mp4",
#   resize_width=640,
#   show_window=True
# )
