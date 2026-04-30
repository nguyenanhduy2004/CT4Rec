# CT4Rec TensorFlow 1.x → 2.x Migration Skill File

## Project Overview

- **Name**: CT4Rec (Consistency-learning based Transformer 4 Recommendation)
- **Type**: Deep learning recommendation system using SASRec (Self-Attentive Sequential Recommendation)
- **Location**: `d:\inclass\4-HK2\Data Mining\CT4Rec`
- **Language**: Python 3.x with TensorFlow
- **Main Entry**: `main.py`

## Original Issue

Project was written for TensorFlow 1.x (gpu==1.12.0) which is no longer suitable. Code uses:

- `tf.Session()` and explicit session management
- `tf.placeholder()` for inputs
- `tf.variable_scope()` and `tf.AUTO_REUSE`
- `tf.contrib` APIs (deprecated)
- `tf.to_float()`, `tf.to_int32()` (deprecated)
- `tf.layers` (partial TF1 API)

## Migration Work Completed ✅

### 1. Created Compatibility Shim (tf_compat.py)

- File: `tf_compat.py` (NEW)
- Purpose: Provides TensorFlow 2.x compatibility layer using `tf.compat.v1` + `disable_v2_behavior()`
- Code:

  ```python
  import types
  import tensorflow as _tf

  if _tf.__version__.startswith('2'):
      import tensorflow.compat.v1 as tf
      tf.disable_v2_behavior()
  else:
      tf = _tf

  if not hasattr(tf, 'random') or not hasattr(getattr(tf, 'random', types.SimpleNamespace()), 'set_random_seed'):
      tf.random = types.SimpleNamespace(set_random_seed=lambda s: getattr(tf, 'set_random_seed')(s))

  __all__ = ['tf']
  ```

- All Python files import from this shim: `from tf_compat import tf`

### 2. Updated Python Files to Use Shim

**Modified imports:**

- `main.py`: Line 1 changed to `from tf_compat import tf`
- `model.py`: Added `from tf_compat import tf` at top
- `modules.py`: Line 11 changed to `from tf_compat import tf`

### 3. Replaced Deprecated APIs

**In modules.py:**

- `layer_norm()` function: Removed `tf.contrib.layers.layer_norm()`, now uses local `normalize()` implementation
- Embedding regularizer: Replaced `tf.contrib.layers.l2_regularizer(l2_reg)` with `tf.keras.regularizers.l2(l2_reg)` (with null check for l2_reg==0)

**In model.py:**

- Line 16: `tf.to_float(mask_bool)` → `tf.cast(mask_bool, tf.float32)`
- Line 82: `tf.to_float(tf.not_equal(...))` → `tf.cast(tf.not_equal(...), tf.float32)`
- Line 205: Replaced deprecated `tf.batch_gather(prob_sim, idx)` with `tf.gather_nd()` pattern:
  ```python
  row_idx = tf.expand_dims(tf.range(tf.shape(prob_sim)[0]), 1)
  full_idx = tf.concat([row_idx, idx], axis=1)
  diag_part_sim = tf.gather_nd(prob_sim, full_idx)
  ```

### 4. Updated Dependencies

- `requirement.txt`: Changed `tensorflow-gpu==1.12.0` to `tensorflow==2.16.1`
- Installed via pip: `tensorflow==2.16.1`, `numpy`, `tqdm` (existing)

### 5. Added Testing & Documentation

- `dev_smoke_test.py` (NEW): Basic import + model graph construction test
- `MIGRATION.md` (NEW): User-friendly run instructions and notes

## Project Structure

```
CT4Rec/
├── main.py                 # Training script (requires dataset partition)
├── model.py                # Model class defining SASRec architecture
├── modules.py              # Helper functions (embedding, attention, feed-forward)
├── sampler.py              # WarpSampler for negative sampling
├── util.py                 # Data loading utilities (data_partition, evaluation functions)
├── tf_compat.py            # NEW: TF2 compatibility shim
├── dev_smoke_test.py       # NEW: Minimal smoke test
├── requirement.txt         # Dependencies (updated)
├── MIGRATION.md            # NEW: Run instructions
├── PROJECT_SKILL.md        # This file - complete project documentation
├── data/
│   └── Beauty.txt          # Sample dataset
└── README.md               # Original project docs
```

## Key Files & What They Do

### main.py

- Entry point for training
- Handles argument parsing, dataset loading, training loop
- Uses `tf.Session()` for session-based execution (compatible via shim)
- Saves checkpoints, logs metrics to `train_dir/log.txt`
- Args: `--dataset`, `--train_dir`, `--batch_size`, `--hidden_units`, `--dropout_rate`, `--con_alpha`, `--rd_alpha`, etc.

### model.py

- Defines `Model` class (graph building in `__init__`)
- Inputs: user IDs, item sequences, positive/negative samples
- Outputs: logits, loss, auc, training op, merged summaries
- Methods:
  - `user_encoder()`: Stacks multi-head self-attention + feed-forward blocks
  - `weight_info_nce()`: Contrastive loss computation
  - `get_softmax_loss()`, `get_r_dropout_loss()`: Loss functions

### modules.py

- Low-level building blocks:
  - `embedding()`: Item/position embeddings with optional L2 regularization
  - `multihead_attention()`: Self-attention with causality masking
  - `feedforward()`: Conv1D feed-forward network
  - `normalize()`: Layer normalization
  - `positional_encoding()`: Positional embeddings for sequence

### sampler.py

- `WarpSampler`: Async negative sampling (multi-worker thread pool)
- Yields batches of (user_id, input_seq, positive_items, negative_items)

### util.py

- `data_partition()`: Loads dataset from file, splits into train/valid/test
- `evaluate()`, `evaluate_valid()`: Compute NDCG@k and HR@k metrics

## Current State & Known Issues

### What Works ✅

- Code compiles without syntax errors
- TensorFlow 2.x imports succeed via shim
- Model graph builds (verified with smoke test, though slow in this environment)
- All deprecated APIs replaced
- tf.contrib removed

### What's Untested ⚠️

- Actual training run (smoke test hangs due to TF init delays in this environment)
- Model convergence / metric computation
- Data loading / sampler integration
- SavedModel checkpoint formats

### Known Limitations

- Still using `tf.compat.v1` (graph-based, eager=False)
- Still using `tf.Session()`, placeholders, variable_scope
- Not a true TF2 Keras model (no subclassing `tf.keras.Model`)
- Slow TF initialization in shared environments (recommend running locally)

## How to Run (From MIGRATION.md)

### Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirement.txt
```

### Optional: Reduce TF startup delay

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:TF_ENABLE_ONEDNN_OPTS = '0'
$env:TF_CPP_MIN_LOG_LEVEL = '2'
```

### Run training

```powershell
python main.py --dataset data --train_dir train_output --batch_size 128 --num_epochs 2
```

### Expected outputs

- `train_output/check_path/`: Saved model checkpoints
- `train_output/log.txt`: Metrics (epoch, time, NDCG@k, HR@k)
- `train_output/summary/`: TensorBoard logs

## Next Steps (Priority Order)

### Option A: Full Native TF2 Migration (Recommended Long-term) 🎯

**Effort**: Medium (2-4 hours)
**Benefits**: Cleaner code, better performance, future-proof
**Steps**:

1. Convert `Model` to subclass `tf.keras.Model`
2. Replace `tf.placeholder()` with function arguments
3. Replace `tf.Session()` with eager execution or `tf.function()`
4. Convert data pipeline to `tf.data.Dataset`
5. Replace `tf.train.Saver` with `tf.keras.Model.save()`
6. Remove `tf_compat.py` dependency

### Option B: Quick Bug Fixes (If issues arise) 🔧

- Check actual training runs; debug any runtime errors
- Adjust sampler, data loading if needed
- Fix metric computation

### Option C: Performance Optimization

- Profile bottlenecks (TF ops, sampler, I/O)
- Optimize batch processing with `tf.data`
- Reduce memory footprint for larger datasets

## References & Important Notes

### TF2 API Migration Resources

- `tf.compat.v1` docs: https://www.tensorflow.org/api_docs/python/tf/compat/v1
- Keras Model API: https://www.tensorflow.org/api_docs/python/tf/keras/Model
- Migration guide: https://www.tensorflow.org/guide/migrate

### Code Patterns Used

- Graph-based (static computation graph)
- Session-based execution
- Variable scoping for parameter sharing
- tf.layers for high-level ops (deprecated but works with compat.v1)

### Dataset Format (data/Beauty.txt)

- Format: user_id,item_id (comma-separated)
- One interaction per line
- Processed by `data_partition()` into sequences

## Checklist for Future Sessions

When resuming work on this project:

- [ ] Check if there are any new runtime errors in `main.py` training
- [ ] Verify TensorFlow version in venv is 2.16.1+
- [ ] Check MIGRATION.md for latest run instructions
- [ ] If migrating to native TF2, start with Option A above
- [ ] Run `dev_smoke_test.py` first to verify imports (may be slow)
- [ ] Test on small dataset/batch before full training
