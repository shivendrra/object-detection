import cv2
import numpy as np

def apply_global_heatmap(frame, diff_mask, colormap=cv2.COLORMAP_JET, alpha=0.4):
  heatmap = cv2.applyColorMap(diff_mask, colormap)
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2BGRA)
  frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
  blended = cv2.addWeighted(frame_bgra, 1 - alpha, heatmap, alpha, 0)
  return cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)

def draw_symbol_mask(frame, mask, symbol, color=(0, 255, 0), font_scale=0.4, thickness=1):
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for cnt in contours:
    if cv2.contourArea(cnt) < 200: continue
    x, y, w, h = cv2.boundingRect(cnt)
    step_x = max(10, w // 10)
    step_y = max(10, h // 10)
    for i in range(x, x + w, step_x):
      for j in range(y, y + h, step_y):
        if cv2.pointPolygonTest(cnt, (i, j), False) >= 0:
          cv2.putText(frame, symbol, (i, j), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def merge_boxes(boxes, overlapThresh=0.3):
  if len(boxes) == 0: return []
  boxes_np = np.array(boxes)
  x1 = boxes_np[:, 0]
  y1 = boxes_np[:, 1]
  x2 = boxes_np[:, 0] + boxes_np[:, 2]
  y2 = boxes_np[:, 1] + boxes_np[:, 3]

  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(y2)
  pick = []
  while len(idxs) > 0:
    last = idxs[-1]
    pick.append(last)
    xx1 = np.maximum(x1[last], x1[idxs[:-1]])
    yy1 = np.maximum(y1[last], y1[idxs[:-1]])
    xx2 = np.minimum(x2[last], x2[idxs[:-1]])
    yy2 = np.minimum(y2[last], y2[idxs[:-1]])
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    overlap = (w * h) / areas[idxs[:-1]]
    idxs = idxs[np.where(overlap <= overlapThresh)]
  return [boxes[i] for i in pick]

# --- SETTINGS ---
VIDEO_IN = "data/input.mp4"
VIDEO_OUT = "data/overlay.mp4"
RESIZE_W = 640

cap = cv2.VideoCapture(VIDEO_IN)
ret, reference_frame = cap.read()
if not ret: raise RuntimeError("Cannot read video")

h0, w0 = reference_frame.shape[:2]
h1 = int(h0 * RESIZE_W / w0)
fps = cap.get(cv2.CAP_PROP_FPS)
reference_frame = cv2.resize(reference_frame, (RESIZE_W, h1))
reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (RESIZE_W, h1))

while True:
  ret, frame = cap.read()
  if not ret: break

  frame = cv2.resize(frame, (RESIZE_W, h1))
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  diff = cv2.absdiff(reference_gray, gray)
  _, binary_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
  opened = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

  contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  raw_boxes = []
  for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 20 and h > 20:
      raw_boxes.append((x, y, w, h))

  merged_boxes = merge_boxes(raw_boxes)
  added_mask = np.zeros_like(gray)
  removed_mask = np.zeros_like(gray)

  for x, y, w, h in merged_boxes:
    region_ref = reference_gray[y:y+h, x:x+w]
    region_now = gray[y:y+h, x:x+w]
    score = np.mean(region_now) - np.mean(region_ref)

    if score > 10:
      # removed
      label = "Missing object"
      color = (0, 0, 255)
      removed_mask[y:y+h, x:x+w] = 255
    else:
      # added
      label = "Added object"
      color = (0, 255, 0)
      added_mask[y:y+h, x:x+w] = 255

    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
  overlay = apply_global_heatmap(frame, diff) # create a global heatmap over the entire frame
  out.write(overlay)
  cv2.imshow("Overlay", overlay)
  if cv2.waitKey(1) & 0xFF == 27:
    break

cap.release()
out.release()
cv2.destroyAllWindows()
