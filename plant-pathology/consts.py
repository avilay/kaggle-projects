from pathlib import Path

RUST = 0
SCAB = 1
HEALTHY = 2
MULTIPLE_DISEASES = 3
# DATAROOT = path.expanduser("~/ml-data/plant-pathology")
DATAROOT = Path.home() / "ml-data" / "plant-pathology"
NUM_CLASSES = 4
LABELS = ["rust", "scab", "healthy", "multiple_diseases"]
# EXPROOT = path.join("~", "temp", "experiments")
EXPROOT = Path.home() / "temp" / "experiments"
