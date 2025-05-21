import cv2
import numpy as np

VIDEO_IN  = "data/input.mp4"
VIDEO_OUT = "data/overlay.mp4"
RESIZE_W  = 1280

cap = cv2.VideoCapture(VIDEO_IN)
ret, frame0 = cap.read()
if not ret:
  raise RuntimeError("Cannot read video")

h0, w0 = frame0.shape[:2]
h1 = int(h0 * RESIZE_W / w0)
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (RESIZE_W, h1))
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=60, detectShadows=True)

while True:
  ret, frame = cap.read()
  if not ret:
    break
  frame = cv2.resize(frame, (RESIZE_W, h1))
  gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  fg_mask = backSub.apply(gray)
  fg_mask[fg_mask == 127] = 0
  _, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
  fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
  heat = cv2.applyColorMap(fg_mask, cv2.COLORMAP_JET)
  overlay = cv2.addWeighted(frame, 0.6, heat, 0.4, 0) # Overlay heatmap on original
  out.write(overlay)
  cv2.imshow("Heatmap Overlay", overlay)
  if cv2.waitKey(1) & 0xFF == 27:
    break

cap.release()
out.release()
cv2.destroyAllWindows()