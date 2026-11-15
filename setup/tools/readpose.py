import sys, json, cv2
import scservo_sdk as scs
out = sys.argv[1]
cal = json.load(open(r"C:\Users\sudar\projects\third-hand\setup\my_follower_calibration_2026-11-14.json"))
p, h = scs.PortHandler("COM3"), scs.PacketHandler(0)
p.openPort(); p.setBaudRate(1_000_000)
def rd(i, a, n=2):
    for _ in range(6):
        v, c, _ = (h.read1ByteTxRx if n == 1 else h.read2ByteTxRx)(p, i, a)
        if c == scs.COMM_SUCCESS: return v
    raise RuntimeError("read failed")
pose = {}
for name, c in cal.items():
    i = c["id"]; raw = rd(i, 56); lo, hi = c["range_min"], c["range_max"]
    deg = (raw - (lo + hi) / 2) * 360 / 4096
    ok = lo <= raw <= hi
    pose[name] = raw
    print(f"{i} {name:13s} raw {raw:4d}  range {lo}-{hi}  {deg:+7.1f} deg  torque {rd(i,40,1)}  {'in range' if ok else 'OUT OF RANGE'}")
p.closePort()

cap = cv2.VideoCapture(2); cap.set(3, 1280); cap.set(4, 720)
f = None
for _ in range(12):
    ok, fr = cap.read()
    if ok: f = fr
cap.release()
cv2.imwrite(out + "/point_pen.jpg", cv2.resize(cv2.rotate(f, cv2.ROTATE_180), (640, 360)))
