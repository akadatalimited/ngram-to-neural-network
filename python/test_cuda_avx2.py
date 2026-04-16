python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

python - <<'PY'
import tensorflow as tf
print("tensorflow:", tf.__version__)
print("gpus:", tf.config.list_physical_devices('GPU'))
PY

nvidia-smi
lscpu | grep -i avx
