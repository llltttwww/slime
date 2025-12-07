export SLIME_ROOT=/mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime
export MEGATRON_ROOT=/mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/Megatron-LM/

source /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/scripts/models/qwen3-kimi-2B-A0.5B.sh

PYTHONPATH=${SLIME_ROOT}:${MEGATRON_ROOT} python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--yuchenFan--Qwen3-Next-Kimi-1204/snapshots/fccadc7499f8a9bcef45f30f409143e01640bef7 \
    --save /mnt/shared-storage-user/p1-shared/luotianwei/checkpoints/Qwen3-Next-Kimi-1204