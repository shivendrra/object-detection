import cv2
import numpy as np
from collections import deque

# Constants
FRAME_BUFFER = 30  # how long to persist the overlay after disappearance
X_SPACING = 15
Y_SPACING = 15
VIDEO_PATH = "data/input.mp4"
OUTPUT_PATH = "data/output.mp4"
MAX_HISTORY = 100 # number of past masks to store
MIN_AREA = 500 # min area to consider as object
RESIZE_WIDTH = 640 # resizing width for faster processing
removed_objects = deque(maxlen=100)
heatmaps = deque(maxlen=100)

# pplying heatmap
def apply_heatmap(diff):
  heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
  return heatmap

# overlaying text and bounding boxes with symbols
def draw_removed_overlay(frame, contours, timestamp):
  overlay = frame.copy()
  for cnt in contours:
    if cv2.contourArea(cnt) < 500:
      continue
    x, y, w, h = cv2.boundingRect(cnt)
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    
    # putting red Xs over masked region
    for yy in range(y, y + h, Y_SPACING):
      for xx in range(x, x + w, X_SPACING):
        if mask[yy, xx] == 255:
          cv2.putText(overlay, 'X', (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(overlay, 'Missing object', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    removed_objects.append((timestamp, overlay.copy()))

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
resize_h = int(frame_h * (RESIZE_WIDTH / frame_w))

ret, prev = cap.read()
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (RESIZE_WIDTH, resize_h))
prev = cv2.resize(prev, (RESIZE_WIDTH, resize_h))  # make sure prev frame is resized too
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
frame_count = 0

while True:
  ret, frame = cap.read()
  if not ret:
    break
  frame = cv2.resize(frame, (RESIZE_WIDTH, resize_h))
  frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  diff = cv2.absdiff(prev_gray, frame_gray)   # Frame differencing
  _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
  dilated = cv2.dilate(thresh, None, iterations=2)

  contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # Get contours of removed objects
  global_heatmap = apply_heatmap(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX))
  heatmaps.append((frame_count, global_heatmap.copy()))
  draw_removed_overlay(frame, contours, frame_count)  # Draw removed object overlays
  combined = cv2.addWeighted(frame, 0.6, global_heatmap, 0.4, 0)    # Combine global heatmap and overlays

  # Overlay the removed object overlays using masks instead of repeated addWeighted
  for t, overlay in removed_objects:
    if frame_count - t < FRAME_BUFFER:
      mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
      mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]
      inv_mask = cv2.bitwise_not(mask)

      fg = cv2.bitwise_and(overlay, overlay, mask=mask)
      bg = cv2.bitwise_and(combined, combined, mask=inv_mask)
      combined = cv2.add(bg, fg)

  out.write(combined)
  cv2.imshow('Change Detection', combined)
  frame_count += 1
  prev_gray = frame_gray.copy()
  if cv2.waitKey(1) & 0xFF == 27:
    break

cap.release()
out.release()
cv2.destroyAllWindows()