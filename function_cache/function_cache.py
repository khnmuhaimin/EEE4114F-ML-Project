import sys
from pathlib import Path
import diskcache as dc

MAIN_DIR = Path(sys.argv[0]).resolve().parent   # dir of the main file
PROJECT_DIR = MAIN_DIR.parent
DEFAULT_CACHE = dc.Cache(str(PROJECT_DIR / "cache"))  # specify cache directory
DEBUG_CACHE = dc.Cache(str(PROJECT_DIR / "debug_cache"))