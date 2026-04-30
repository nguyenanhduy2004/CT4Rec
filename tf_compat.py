import os
import types

# Keras 3 removes tf.layers APIs; force legacy Keras for TF1-style code paths.
if 'TF_USE_LEGACY_KERAS' not in os.environ:
    os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as _tf

if _tf.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
else:
    tf = _tf

# Ensure tf.random.set_random_seed is available for code that calls it
if not hasattr(tf, 'random') or not hasattr(getattr(tf, 'random', types.SimpleNamespace()), 'set_random_seed'):
    tf.random = types.SimpleNamespace(set_random_seed=lambda s: getattr(tf, 'set_random_seed')(s))

__all__ = ['tf']
