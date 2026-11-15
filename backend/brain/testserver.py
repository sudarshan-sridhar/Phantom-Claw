"""A second copy of the agent server on its own port, with the pretend arm, for trying the brain
route end to end without the robot. The real app (port 8765) and the real arm are not touched.

    python -m backend.brain.testserver [port]        (default 8775)

Same agents, same models, same plan check as the real app. The arm is the pretend one with every
grip height counted as taught, laptop tools only say what they would do, and nothing is spoken.
"""
import os
import sys


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8775
    os.environ.pop("PHANTOM_ARM", None)                 # never the real arm from here
    os.environ.setdefault("PHANTOM_FAKE_TIERS", "tall,can,low")
    os.environ.setdefault("PHANTOM_DRY_TOOLS", "1")
    os.environ.setdefault("PHANTOM_NO_BROWSER", "1")

    from backend.agents import config
    config.PORT = port
    config.PUBLIC_BASE = f"http://{config.HOST}:{port}"

    import uvicorn
    from backend import server
    from backend.agents.pipeline import Pipeline

    # same pipeline as the app, with speech off so no voice clips are made for a test
    server.state["pipeline"] = Pipeline(server.get_arm(), server.hub.emit, speak=False)
    uvicorn.run(server.app, host=config.HOST, port=port, log_level="warning")


if __name__ == "__main__":
    main()
