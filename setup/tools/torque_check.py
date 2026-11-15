"""Exit quietly if torque is on for the four arm joints; print AssertionError if not."""
import scservo_sdk as scs
p, h = scs.PortHandler("COM3"), scs.PacketHandler(0)
p.openPort(); p.setBaudRate(1_000_000)
on = []
for i in (1, 2, 3, 4):
    for _ in range(6):
        v, c, _ = h.read1ByteTxRx(p, i, 40)
        if c == scs.COMM_SUCCESS:
            on.append(v); break
p.closePort()
print("torque on" if on == [1, 1, 1, 1] else "AssertionError: torque is off")
