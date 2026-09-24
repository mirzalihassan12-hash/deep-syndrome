"""
Makes the project's Python source (inference.py, face_crop.py, training/)
importable from this test-cases folder, which lives at the project root
alongside them:

    deep-syndrome/
      inference.py
      face_crop.py
      training/
      test-cases/
        python/
          conftest.py      <- this file

Override with the PROJECT_ROOT environment variable if you've moved this
folder elsewhere:
    PROJECT_ROOT=/path/to/deep-syndrome pytest
"""
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CANDIDATES = [
    os.environ.get("PROJECT_ROOT"),
    THIS_DIR.parent.parent,  # test-cases/python -> test-cases -> project root
]

for candidate in CANDIDATES:
    if not candidate:
        continue
    candidate = Path(candidate)
    if (candidate / "inference.py").exists():
        sys.path.insert(0, str(candidate))
        break
else:
    raise RuntimeError(
        "Could not find the DeepSyndrome project source. Set PROJECT_ROOT to "
        "its path, e.g.: PROJECT_ROOT=/path/to/deep-syndrome pytest"
    )
