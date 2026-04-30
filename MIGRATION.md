Migration and run notes

What I changed

- Added `tf_compat.py` which imports `tensorflow.compat.v1` and calls `disable_v2_behavior()` when TF2 is installed. Import `tf` from this shim in code.
- Replaced `tf.contrib.layers.l2_regularizer` with `tf.keras.regularizers.l2`.
- Replaced `tf.contrib.layers.layer_norm` usage with the local `normalize()` implementation.
- Replaced deprecated `tf.to_float` with `tf.cast(..., tf.float32)` and replaced `tf.batch_gather` with a `gather_nd` pattern.
- Updated `requirement.txt` to recommend `tensorflow==2.16.1`.

How to run (recommended)

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirement.txt
```

2. To reduce TensorFlow startup delays (optional), set these env vars before running:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:TF_ENABLE_ONEDNN_OPTS = '0'
$env:TF_CPP_MIN_LOG_LEVEL = '2'
```

3. Run the training (example):

```powershell
python main.py --dataset data --train_dir train_output --batch_size 128 --num_epochs 2
```

Notes and next steps

- I attempted to run a smoke-test here, but TensorFlow initialization in this environment caused long startup times; you may need to run the smoke test locally (see env vars above).
- I can proceed to fully migrate the code to native TF2 (`tf.keras`, `tf.data`, `tf.function`) which will remove reliance on `tf.compat.v1`. Tell me if you want that.
