import cv2
import numpy as np

def apply_transparent_heatmap(frame, mask, colormap=cv2.COLORMAP_JET, alpha=0.3):
  heatmap = cv2.applyColorMap(mask, colormap)
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2BGRA)
  overlay = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
  mask_bool = mask > 0  # ensure mask is 2D and converted to boolean
  mask_expanded = np.repeat(mask_bool[:, :, np.newaxis], 4, axis=2) # expanding mask to 4 channels for proper alpha blending

  overlay[mask_expanded] = (
    (1 - alpha) * overlay[mask_expanded] + alpha * heatmap[mask_expanded]
  ).astype(np.uint8)  # applying weighted blend only on masked regions
  return cv2.cvtColor(overlay, cv2.COLOR_BGRA2BGR)

def merge_boxes(boxes, overlapThresh=0.3):
  if len(boxes) == 0:
    return []

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

VIDEO_IN = "data/input.mp4"
VIDEO_OUT = "data/overlay.mp4"
RESIZE_W = 640

cap = cv2.VideoCapture(VIDEO_IN)
ret, reference_frame = cap.read()
if not ret:
  raise RuntimeError("Cannot read video")

h0, w0 = reference_frame.shape[:2]
h1 = int(h0 * RESIZE_W / w0)
fps = cap.get(cv2.CAP_PROP_FPS)
reference_frame = cv2.resize(reference_frame, (RESIZE_W, h1))
reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (RESIZE_W, h1))

while True:
  ret, frame = cap.read()
  if not ret:
    break

  frame = cv2.resize(frame, (RESIZE_W, h1))
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  diff = cv2.absdiff(reference_gray, gray)
  _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  raw_boxes = []
  for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 20 and h > 20:
      raw_boxes.append((x, y, w, h))

  merged_boxes = merge_boxes(raw_boxes)
  mask = np.zeros_like(gray)

  for x, y, w, h in merged_boxes:
    region_ref = reference_gray[y:y+h, x:x+w]
    region_now = gray[y:y+h, x:x+w]
    score = np.mean(region_now) - np.mean(region_ref)

    # had to swap these coz it represents the new empty space added/removed
    # so for object it should be reversed.
    if score > 10:
      label = "Removed"
      color = (0, 0, 255)
    else:
      label = "Added"
      color = (0, 255, 0)
      mask[y:y+h, x:x+w] = 255

    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

  overlay = apply_transparent_heatmap(frame, mask)
  out.write(overlay)
  cv2.imshow("Overlay", overlay)
  if cv2.waitKey(1) & 0xFF == 27:
    break

cap.release()
out.release()
cv2.destroyAllWindows()