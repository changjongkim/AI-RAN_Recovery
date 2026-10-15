#!/usr/bin/env bash
# Environment of the recovery project.  Source it on a GPU node (the queue runner does).
# Large assets are read from airan_cloudlab (read-only use): the Aerial repository and image, the TensorRT engines and
# the TensorFlow/Sionna packages.  Everything this project writes stays under $REC.

REC=/pscratch/sd/s/sgkim/kcj/airan_recovery
ANT=/pscratch/sd/s/sgkim/kcj/airan_cloudlab
AERIAL_IMAGE=nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
AERIAL_REPO=$ANT/third_party/aerial-cuda-accelerated-ran
RUNTIME=$ANT/runtime/softwall_same_gpu
TF_DEPS=$RUNTIME/nrx_export_deps
NRX_REPO=$REC/third_party/neural_rx
DATA=$REC/data
RESULTS=$REC/results
ENGINE_LARGE=/runtime/engines/nv/nrx_large_273prb_2ue.trt     # paths inside the container
ENGINE_SMALL=/runtime/engines/nv/nrx_rt_273prb_2ue.trt
GPU_LD_PATH=/opt/udiImage/modules/gpu/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib

# Runtime receivers (cuPHY, pyAerial, TensorRT).  Usage: GPU=n rx python3 /rec/decode_harq.py ...
rx() {
    shifter --module=gpu --image="$AERIAL_IMAGE" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$REC/src:/rec" --volume="$RUNTIME:/runtime" \
        --env=LD_LIBRARY_PATH="$GPU_LD_PATH" --env=PYTHONPATH=/rec:/rec/vendor:/opt/nvidia/cuBB/pyaerial/src \
        --env=CUDA_MODULE_LOADING=LAZY --env=CUDA_VISIBLE_DEVICES="${GPU:-0}" "$@"
}

# Link-level generation (TensorFlow, Sionna, the NVlabs neural_rx repository).  Usage: tfgen python3 $REC/src/gen_harq.py ...
tfgen() {
    shifter --module=gpu --image="$AERIAL_IMAGE" --env=PYTHONPATH="$TF_DEPS" --env=PYTHONNOUSERSITE=1 "$@"
}
