import sys, time, cv2
cap = cv2.VideoCapture(2); cap.set(3, 1280); cap.set(4, 720)
ok, f = cap.read()
h, w = f.shape[:2]
vw = cv2.VideoWriter(sys.argv[1], cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
t = time.time()
while time.time() - t < float(sys.argv[2]):
    ok, f = cap.read()
    if ok: vw.write(cv2.rotate(f, cv2.ROTATE_180))
vw.release(); cap.release()
