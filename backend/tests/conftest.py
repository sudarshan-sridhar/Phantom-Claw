"""Shared test setup."""
import os

# The app now lets one character speak per request. The older tests were written for the
# talkative mode (Merlin's hand-on line and the argument), so they run with it switched on.
os.environ.setdefault("PHANTOM_CHATTER", "1")
