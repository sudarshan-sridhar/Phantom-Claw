"""Brain signal reader for the PiEEG Octopus 16: eyes-closed STOP and a focus value."""
import os
from pathlib import Path

# liblsl prints every network interface at start-up. Point it at our own config, which
# turns its log down to errors only, before anything imports pylsl.
os.environ.setdefault("LSLAPICFG", str(Path(__file__).with_name("lsl_api.cfg")))
