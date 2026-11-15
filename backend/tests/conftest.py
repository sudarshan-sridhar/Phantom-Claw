"""Shared test setup."""
import os

# The app now lets one character speak per request. The older tests were written for the
# talkative mode (Merlin's hand-on line and the argument), so they run with it switched on.
os.environ.setdefault("PHANTOM_CHATTER", "1")

# The listener now acts only when a name is said. The older listener tests were written for
# the earlier default, so they run with it off unless they ask otherwise.
os.environ.setdefault("LISTEN_REQUIRE_NAME", "0")
