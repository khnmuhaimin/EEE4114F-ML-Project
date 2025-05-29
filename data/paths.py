import sys
from pathlib import Path

main_dir = Path(sys.argv[0]).resolve().parent   # dir of the main file
motion_sense_path = main_dir.parent / "motion-sense"  # dir of motion sense repo