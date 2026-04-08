import sys
import os
from pathlib import Path

# Fix: Ensure 'src' is in sys.path so we can find 'adaptive_alert_triage' and 'openenv_shim'
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from adaptive_alert_triage.server import app

import uvicorn
import os

def main():
    """Main entry point for the server application."""
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
