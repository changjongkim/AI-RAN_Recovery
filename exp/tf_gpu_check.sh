#!/usr/bin/env bash
# TensorFlow sees the GPUs, and a small matrix product runs on one.
source /pscratch/sd/s/sgkim/kcj/airan_recovery/env.sh
tfgen python3 -c "
import os; os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
g = tf.config.list_physical_devices('GPU'); print('GPUs:', g)
with tf.device('/GPU:0'):
    x = tf.random.normal([2048, 2048]); y = tf.matmul(x, x)
print('matmul on', y.device, float(tf.reduce_sum(y)))
import sys; sys.exit(0 if g else 1)
"
