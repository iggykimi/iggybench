# common.sh - shared defaults, sourced by the other scripts. Override any of them from the environment.
# shellcheck shell=bash disable=SC2034
BENCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IGGY_HOME=${IGGY_HOME:-$HOME/iggybench}          # models, datasets, caches and logs live here
MODELS_DIR=${MODELS_DIR:-$IGGY_HOME/models}
DATASETS_DIR=${DATASETS_DIR:-$IGGY_HOME/datasets}
PYLIB_DIR=${PYLIB_DIR:-$IGGY_HOME/pylib}         # Python source trees mounted into containers (diffusers, laya)
CACHE_DIR=${CACHE_DIR:-$IGGY_HOME/cache}         # vLLM compile cache and FlashInfer/Triton JIT kernels
LOG_DIR=${LOG_DIR:-$IGGY_HOME/logs}
RESULTS_DIR=${RESULTS_DIR:-$BENCH/results}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
DOCKER=${DOCKER:-docker}                         # the command that runs docker for your user (word-split)
VLLM_IMG=${VLLM_IMG:-vllm/vllm-openai:v0.32.0}   # engine image, and the image that runs `vllm bench serve`
GPU_FIELDS=timestamp,power.draw.instant,utilization.gpu,memory.used,temperature.gpu,clocks.sm,module.power.draw.instant

# Docker container name for a tag ('+' and other characters are not allowed in names)
container_name() { local t=${1//[^a-zA-Z0-9_.-]/-}; echo "iggy-$t"; }
