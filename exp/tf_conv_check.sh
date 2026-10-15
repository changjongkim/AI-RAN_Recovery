#!/usr/bin/env bash
# Which TensorFlow setting trains a separable-convolution model on the GPU (the N4 training failed in
# xla.gpu.conv.backward.filter with CUDNN_STATUS_EXECUTION_FAILED).  Prints one line per variant.
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
cat > /tmp/conv_check_$$.py <<'PY'
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
jit = sys.argv[1] == "jit"
m = tf.keras.Sequential([tf.keras.layers.SeparableConv2D(128, 3, padding="same", activation="relu"),
                         tf.keras.layers.SeparableConv2D(128, 3, padding="same", activation="relu"),
                         tf.keras.layers.Conv2D(8, 1)])
opt = tf.keras.optimizers.Adam(1e-4)
@tf.function(jit_compile=jit)
def step(x):
    with tf.GradientTape() as t:
        loss = tf.reduce_mean(tf.square(m(x)))
    g = t.gradient(loss, m.trainable_weights)
    opt.apply_gradients(zip(g, m.trainable_weights))
    return loss
x = tf.random.normal([64, 48, 14, 40])
try:
    for _ in range(20):
        l = step(x)
    print(f"OK loss {float(l):.4f}")
except Exception as e:
    print("FAIL", type(e).__name__, str(e).splitlines()[0][:160])
PY
for v in "jit|" "nojit|" "jit|--xla_gpu_autotune_level=0" "jit|--xla_gpu_strict_conv_algorithm_picker=false" "nojit|TF_CUDNN_USE_AUTOTUNE=0"; do
  mode=${v%%|*}; extra=${v#*|}
  xf="--xla_gpu_cuda_data_dir=/usr/local/cuda"; envs=""
  case $extra in --*) xf="$xf $extra" ;; TF_*) envs="--env=$extra" ;; esac
  r=$(shifter --module=gpu --image="$AERIAL_IMAGE" --env=PYTHONPATH="$TF_DEPS" --env=PYTHONNOUSERSITE=1 \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH:$CUDNN8" --env=XLA_FLAGS="$xf" --env=TF_CPP_MIN_LOG_LEVEL=3 $envs \
        --env=PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin python3 /tmp/conv_check_$$.py $mode 2>&1 | grep -E "^OK|^FAIL" | tail -1)
  echo "$mode $extra: $r"
done
rm -f /tmp/conv_check_$$.py
