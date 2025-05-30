import cv2
import numpy as np
import threading
import queue, time
from typing import Union

def preprocess_frame(frame, size):
  frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  return frame, gray

def apply_global_heatmap_cuda(frame, mask, alpha=0.4):
  if cv2.countNonZero(mask) == 0:
    return frame
  heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
  return cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)

def merge_boxes(boxes, overlapThresh=0.3):
  if not boxes: return []
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

def worker(input_q, output_q, reference_gray, resize_dim, stop_event, use_cuda):
  kernel = np.ones((5, 5), np.uint8)
  while not stop_event.is_set():
    item = input_q.get()
    if item is None: break
    frame_id, frame = item
    frame, gray = preprocess_frame(frame, resize_dim)

    if use_cuda:
      d_ref = cv2.cuda_GpuMat()
      d_now = cv2.cuda_GpuMat()
      d_ref.upload(reference_gray)
      d_now.upload(gray)
      d_diff = cv2.cuda.absdiff(d_ref, d_now)
      diff = d_diff.download()
    else:
      diff = cv2.absdiff(reference_gray, gray)

    _, binary_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    opened = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = [(x, y, w, h) for cnt in contours
                 if cv2.contourArea(cnt) > 400
                 for x, y, w, h in [cv2.boundingRect(cnt)]]
    merged_boxes = merge_boxes(raw_boxes)

    added_mask = np.zeros_like(gray)
    removed_mask = np.zeros_like(gray)

    for x, y, w, h in merged_boxes:
      region_ref = reference_gray[y:y+h, x:x+w]
      region_now = gray[y:y+h, x:x+w]
      mean_diff = np.mean(region_now) - np.mean(region_ref)

      if abs(mean_diff) < 5:
        continue
      if mean_diff > 10:
        removed_mask[y:y+h, x:x+w] = 255
        label = "Missing"
        color = (0, 0, 255)
      else:
        added_mask[y:y+h, x:x+w] = 255
        label = "Added"
        color = (0, 255, 0)

      cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
      cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    heat_mask = cv2.bitwise_or(added_mask, removed_mask)
    overlay = apply_global_heatmap_cuda(frame, heat_mask)
    output_q.put((frame_id, overlay))

def run_gpu_motion_overlay(input_path: Union[str, int], output_path: str, resize_width=640, show_window=True, use_cuda_if_available=True):
  use_cuda = use_cuda_if_available and cv2.cuda.getCudaEnabledDeviceCount() > 0
  cap = cv2.VideoCapture(input_path)
  ret, ref_frame = cap.read()
  if not ret:
    raise RuntimeError("Cannot read input video")

  h0, w0 = ref_frame.shape[:2]
  h1 = int(h0 * resize_width / w0)
  resize_dim = (resize_width, h1)
  ref_frame, reference_gray = preprocess_frame(ref_frame, resize_dim)

  fps = cap.get(cv2.CAP_PROP_FPS) or 30
  out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, resize_dim)

  input_q = queue.Queue(maxsize=10)
  output_q = queue.Queue()
  stop_event = threading.Event()

  worker_thread = threading.Thread(target=worker, args=(input_q, output_q, reference_gray, resize_dim, stop_event, use_cuda))
  worker_thread.start()

  frame_id = 0
  output_buffer = {}
  fps_display = 0.0
  last_time = time.time()

  while True:
    ret, frame = cap.read()
    if not ret: break
    if not input_q.full():
      input_q.put((frame_id, frame))

    while not output_q.empty():
      fid, processed = output_q.get()
      output_buffer[fid] = processed

    now = time.time()
    delta = now - last_time
    last_time = now
    fps_current = 1.0 / delta if delta > 0 else 0.0
    fps_display = 0.9 * fps_display + 0.1 * fps_current

    while frame_id in output_buffer:
      overlay = output_buffer[frame_id]
      cv2.putText(overlay, f"FPS: {fps_display:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

      if show_window:
        cv2.imshow("Overlay", overlay)
        if cv2.waitKey(1) & 0xFF == 27:
          stop_event.set()
          input_q.put(None)
          cap.release()
          out.release()
          cv2.destroyAllWindows()
          return

      out.write(overlay)
      del overlay
      frame_id += 1

  input_q.put(None)
  stop_event.set()
  worker_thread.join()
  cap.release()
  out.release()
  if show_window:
    cv2.destroyAllWindows()