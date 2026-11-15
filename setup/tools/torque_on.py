"""Turn torque on at the current positions (nothing moves). Refuses if a joint is out of range."""
import json
import scservo_sdk as scs
cal = json.load(open(r"C:\Users\sudar\projects\third-hand\setup\my_follower_calibration_2026-11-14.json"))
RNG = {c["id"]: (c["range_min"], c["range_max"]) for c in cal.values()}
TORQUE = {1: 500, 2: 1000, 3: 800, 4: 500, 5: 400, 6: 300}
p, h = scs.PortHandler("COM3"), scs.PacketHandler(0)
p.openPort(); p.setBaudRate(1_000_000)
def rd(i, a, n=2):
    for _ in range(6):
        v, c, _ = (h.read1ByteTxRx if n == 1 else h.read2ByteTxRx)(p, i, a)
        if c == scs.COMM_SUCCESS: return v
    raise RuntimeError("read failed")
def wr(i, a, v, n=2):
    for _ in range(6):
        c, _ = (h.write1ByteTxRx if n == 1 else h.write2ByteTxRx)(p, i, a, v)
        if c == scs.COMM_SUCCESS: return
    raise RuntimeError("write failed")
start = {i: rd(i, 56) for i in range(1, 7)}
bad = [i for i in (1, 2, 3, 4, 6) if not (RNG[i][0] - 25 <= start[i] <= RNG[i][1] + 25)]
assert not bad, f"joints {bad} out of range: {start}"
for i in range(1, 7):
    g = max(RNG[i][0], min(RNG[i][1], start[i]))
    wr(i, 42, g); assert rd(i, 42) == g
    wr(i, 41, 12, 1); wr(i, 46, 200); wr(i, 48, TORQUE[i])
for i in range(1, 7):
    wr(i, 40, 1, 1)
print("torque on at", start)
p.closePort()
