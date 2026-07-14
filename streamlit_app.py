# Hosted entry point for Streamlit Community Cloud. Sets the US book as the default
# so the app runs with no launch flags, then hands off to the real dashboard.
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("BEMO_DB", str(ROOT / "data" / "bemo_us.duckdb"))
os.environ.setdefault("BEMO_CONFIG", str(ROOT / "config" / "bemo_universe_us.yaml"))
os.environ.setdefault("BEMO_CCY", "USD")
os.environ.setdefault("BEMO_LAB", "1")

import runpy

runpy.run_path(str(ROOT / "reporting" / "dashboard.py"), run_name="__main__")
