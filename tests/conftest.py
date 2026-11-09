import os

# Set MOCK=1 by default for automated test runs
os.environ["MOCK"] = "1"

# Invalidate and reload config if already imported
import sys
if "app.config" in sys.modules:
    import importlib
    import app.config
    importlib.reload(app.config)
    for mod in ["app.tts", "app.stt", "app.llm", "app.main", "app.session", "app.fillers"]:
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])

