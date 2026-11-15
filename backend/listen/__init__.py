"""Voice control for Phantom Claw.

A separate process (`python -m backend.listen`) that owns the microphone, cuts speech into
utterances, turns them into text on this laptop and talks to the server over HTTP.
"""
