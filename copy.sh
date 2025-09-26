# 1) 同步/缓存 JAX ckpt（任选一种）
# A. 让 openpi 自动从 GCS 缓存
/home/jz97/VLM_REPO/openpi/examples/libero/.venv/bin/python - <<'PY'
from openpi.shared import download
print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero"))
PY

# B. 或自己拉到本地（若装了 gsutil）
# gsutil -m rsync -r gs://openpi-assets/checkpoints/pi05_libero /home/jz97/VLM_REPO/openpi/ckpts/pi05_libero_jax

# 2) JAX -> PyTorch 转换
/home/jz97/VLM_REPO/openpi/examples/libero/.venv/bin/python \
  examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir "$( /home/jz97/VLM_REPO/openpi/examples/libero/.venv/bin/python - <<'PY'
from openpi.shared import download
print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero"))
PY
)" \
  --config_name pi05_libero \
  --output_path /home/jz97/VLM_REPO/openpi/ckpts/pi05_libero_torch

# 3) 快速加载自检（PyTorch 路径）
/home/jz97/VLM_REPO/openpi/examples/libero/.venv/bin/python - <<'PY'
from openpi.training import config as _config
from openpi.policies import policy_config
cfg = _config.get_config("pi05_libero")
policy = policy_config.create_trained_policy(cfg, "/home/jz97/VLM_REPO/openpi/ckpts/pi05_libero_torch")
print("Loaded OK (PyTorch).")
PY

# 4) DuQuant 假量化 dry-run（只扫 DiT 主干）
OPENPI_DUQUANT_SCOPE="policy.dit." \
OPENPI_DUQUANT_DRYRUN=1 \
/home/jz97/VLM_REPO/openpi/examples/libero/.venv/bin/python \
  scripts/serve_policy.py --env LIBERO --policy.dir /home/jz97/VLM_REPO/openpi/ckpts/pi05_libero_torch
