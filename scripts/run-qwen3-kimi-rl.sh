#!/bin/bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-kimi-2B-A0.5B.sh"

CKPT_ARGS=(
   --hf-checkpoint /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/checkpoints/qwen-3-next-kimi-2B-A0.5B/qwen3-next-kimi-2B-A0.5B-sft-iter_0000173
   --ref-load /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/checkpoints/qwen-3-next-kimi-2B-A0.5B
   --load /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/checkpoints/qwen-3-kimi-2B-A0.5B-pretrain-kimi-rl
   --save /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/checkpoints/qwen-3-kimi-2B-A0.5B-pretrain-kimi-rl
   --save-interval 100
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/examples/reproducibility/gsm8k/train.parquet
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 100
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 0.8

   --global-batch-size 256
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data gsm8k /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/examples/reproducibility/gsm8k/test.parquet
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 1024
   --eval-top-k 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project qwen3-gsm8k-rl
   --wandb-group "slime-dev" # Use the dynamic name for the specific run
   --wandb-key c30453f4f8daca82665d30e3c2d99fbb9cbf52c0
   --wandb-mode offline
)

SGLANG_JSON_OVERRIDE='{"full_attention_interval":4}'

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7

   --sglang-enable-deterministic-inference
   --sglang-attention-backend flashinfer
   --sglang-json-model-override-args $SGLANG_JSON_OVERRIDE

   --deterministic-mode
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   --calculate-per-token-loss \
   --use-slime-router \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
