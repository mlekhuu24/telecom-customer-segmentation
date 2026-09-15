import os
import sys
from pathlib import Path

inner_dir = Path(__file__).resolve().parent / "Telecom_Customer_Segmentation_Sklearn"
if inner_dir.exists():
    os.chdir(inner_dir)
    sys.path.insert(0, str(inner_dir))
    with open(inner_dir / "predict.py", encoding="utf-8") as f:
        code = f.read()
    exec(code)
