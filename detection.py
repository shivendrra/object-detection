import cv2
import numpy as np
from typing import Union

def apply_global_heatmap(frame, diff_mask, colormap=cv2.COLORMAP_JET, alpha=0.4):
  heatmap = cv2.applyColorMap(diff_mask, colormap)
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2BGRA)
  frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
  blended = cv2.addWeighted(frame_bgra, 1 - alpha, heatmap, alpha, 0)
  return cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)

def draw_symbol_mask(frame, mask, symbol, color=(0, 255, 0), font_scale=0.4, thickness=1):
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 200: continue
    x, y, w, h = cv2.boundingRect(cnt)
    step_x = max(10, w // 10)
    step_y = max(10, h // 10)
    for i in range(x, x + w, step_x):
      for j in range(y, y + h, step_y):
        if cv2.pointPolygonTest(cnt, (i, j), False) >= 0:
          cv2.putText(frame, symbol, (i, j), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def merge_boxes(boxes, overlapThresh=0.3):
  if len(boxes) == 0: return []
  
  # pre-allocating arrays for better performance
  boxes_array = np.array(boxes, dtype=np.float32)
  x1 = boxes_array[:, 0]
  y1 = boxes_array[:, 1]
  x2 = boxes_array[:, 0] + boxes_array[:, 2]
  y2 = boxes_array[:, 1] + boxes_array[:, 3]

  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(y2)
  pick = []

  while len(idxs) > 0:
    last = idxs[-1]
    pick.append(last)
    
    # vectorized computation
    xx1 = np.maximum(x1[last], x1[idxs[:-1]])
    yy1 = np.maximum(y1[last], y1[idxs[:-1]])
    xx2 = np.minimum(x2[last], x2[idxs[:-1]])
    yy2 = np.minimum(y2[last], y2[idxs[:-1]])

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    overlap = (w * h) / areas[idxs[:-1]]
    idxs = idxs[np.where(overlap <= overlapThresh)]
  return [boxes[i] for i in pick]

def run_motion_overlay(input_source: Union[str, int], output_path: str, resize_width=640, alpha=0.4, show_window=True, real_time=True):
  cap = cv2.VideoCapture(input_source)
  cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Optimize video capture settings
  ret, frame = cap.read()
  if not ret:
    raise RuntimeError("Cannot read input")

  h0, w0 = frame.shape[:2]
  h1 = int(h0 * resize_width / w0)
  fps = cap.get(cv2.CAP_PROP_FPS) or 30

  # Pre-compute resize once
  target_size = (resize_width, h1)
  frame = cv2.resize(frame, target_size)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  if real_time:
    background = gray.astype(np.float32)  # Using float32 for better performance
  else:
    reference_gray = gray.copy()

  # pre-allocating morphological kernel
  morph_kernel = np.ones((5, 5), np.uint8)
  
  # Pre-allocating arrays for reuse
  diff_buffer = np.empty_like(gray)
  binary_buffer = np.empty_like(gray)
  
  out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, target_size)

  # Pre-allocating masks
  added_mask = np.zeros_like(gray)
  removed_mask = np.zeros_like(gray)

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    # Resize in-place if possible
    frame = cv2.resize(frame, target_size)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if real_time:
      cv2.accumulateWeighted(gray, background, 0.02)
      ref_frame = cv2.convertScaleAbs(background)
    else:
      ref_frame = reference_gray

    # Use pre-allocated buffer for difference
    cv2.absdiff(ref_frame, gray, diff_buffer)
    
    # Optimized thresholding
    _, binary_diff = cv2.threshold(diff_buffer, 30, 255, cv2.THRESH_BINARY)
    
    # Single morphological operation instead of separate open/close
    opened = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, morph_kernel)

    # Optimized contour finding with less precision for speed
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Pre-filtering contours by area before bounding rect computation
    raw_boxes = []
    for cnt in contours:
      # Quick area check first
      if len(cnt) < 5:  # Skip very small contours
        continue
      x, y, w, h = cv2.boundingRect(cnt)
      if w > 20 and h > 20:
        raw_boxes.append((x, y, w, h))

    merged_boxes = merge_boxes(raw_boxes)
    
    # Clear masks efficiently
    added_mask.fill(0)
    removed_mask.fill(0)

    # Vectorized region analysis where possible
    for x, y, w, h in merged_boxes:
      # Use slicing for faster region extraction
      region_ref = ref_frame[y:y+h, x:x+w]
      region_now = gray[y:y+h, x:x+w]
      
      # Optimized mean calculation
      score = np.mean(region_now, dtype=np.float32) - np.mean(region_ref, dtype=np.float32)

      if score > 10:
        label = "Missing object"
        color = (0, 0, 255)
        removed_mask[y:y+h, x:x+w] = 255
      else:
        label = "Added object"
        color = (0, 255, 0)
        added_mask[y:y+h, x:x+w] = 255

      # Drawing rectangle and text in one go
      cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
      cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Apply heatmap overlay
    overlay = apply_global_heatmap(frame, diff_buffer, alpha=alpha)
    out.write(overlay)

    if show_window:
      cv2.imshow("Overlay", overlay)
      if cv2.waitKey(1) & 0xFF == 27:
        break

  cap.release()
  out.release()
  if show_window:
    cv2.destroyAllWindows()