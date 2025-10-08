#!/usr/bin/env bash
set -euo pipefail

python -c "import torch,sys;print('torch', torch.__version__)" || exit 1
pip install -r requirements-bitblas.txt || true
python - <<'PY'
import os
import subprocess
import sys

try:
    import bitblas  # type: ignore
    print("BitBLAS OK", getattr(bitblas, "__version__", "unknown"))
except Exception as exc:
    repo = "https://github.com/microsoft/BitBLAS"
    print("Falling back to source install from", repo, "due to", repr(exc))
    os.makedirs("third_party", exist_ok=True)
    repo_dir = os.path.join("third_party", "BitBLAS")
    if not os.path.exists(repo_dir):
        subprocess.check_call(["git", "clone", repo, repo_dir], stdout=sys.stdout, stderr=sys.stderr)
    else:
        print("Repository already cloned at", repo_dir)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."], cwd=repo_dir)
PY
