import cv2
import numpy as np
from collections import deque

VIDEO_PATH = "data/input.mp4"
OUTPUT_PATH = "data/output.mp4"
MAX_HISTORY = 100 # number of past masks to store
MIN_AREA = 500 # min area to consider as object
RESIZE_WIDTH = 640 # resizing width for faster processing

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
resize_h = int(frame_h * (RESIZE_WIDTH / frame_w))

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (RESIZE_WIDTH, resize_h))
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)   # bg subtract
mask_history = deque(maxlen=MAX_HISTORY)  # history buffers

while True:
  ret, frame = cap.read()
  if not ret:
    break

  frame = cv2.resize(frame, (RESIZE_WIDTH, resize_h))
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # foreground mask
  fg_mask = bg_subtractor.apply(gray)
  fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
  fg_mask = cv2.dilate(fg_mask, np.ones((3, 3), np.uint8), iterations=2)

  # saving current mask to history
  current_mask = fg_mask.copy()
  mask_history.append(current_mask)

  # detecting contours in current mask
  contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  current_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > MIN_AREA]

  # comparing with previous masks
  removed_boxes = []
  new_boxes = []

  if len(mask_history) >= MAX_HISTORY:
    past_mask = mask_history[0]
    diff_removed = cv2.subtract(past_mask, current_mask)
    diff_new = cv2.subtract(current_mask, past_mask)

    # removed object
    removed_cnts, _ = cv2.findContours(diff_removed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in removed_cnts:
      if cv2.contourArea(c) > MIN_AREA:
        x, y, w, h = cv2.boundingRect(c)
        removed_boxes.append((x, y, w, h))

    # new objects
    new_cnts, _ = cv2.findContours(diff_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in new_cnts:
      if cv2.contourArea(c) > MIN_AREA:
        x, y, w, h = cv2.boundingRect(c)
        new_boxes.append((x, y, w, h))

  # draw boxes
  for (x, y, w, h) in removed_boxes:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(frame, "Removed", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

  for (x, y, w, h) in new_boxes:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, "New", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

  # write and show
  out.write(frame)
  cv2.imshow("Detected Changes", frame)
  if cv2.waitKey(1) & 0xFF == 27:
    break

cap.release()
out.release()
cv2.destroyAllWindows()