"""Run calibrate_arm.py with beeps: high = start, double = go to the finishing end and hold, low = stop."""
import builtins, runpy, sys, threading, winsound
secs = float(sys.argv[sys.argv.index("--secs") + 1])
_p = builtins.print
def warn():
    winsound.Beep(900, 200); winsound.Beep(900, 200)
def cue(*a, **k):
    _p(*a, **k)
    t = " ".join(str(x) for x in a)
    if t.startswith("RECORDING"):
        winsound.Beep(1200, 700)
        threading.Timer(secs - 7, warn).start()
    elif t.startswith("done,"):
        winsound.Beep(500, 900)
builtins.print = cue
sys.argv = ["calibrate_arm.py"] + sys.argv[1:]
runpy.run_path(r"C:\Users\sudar\projects\so101\calibrate_arm.py", run_name="__main__")
