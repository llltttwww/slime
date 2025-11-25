# source /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/scripts/models/qwen3-next-2B-A0.5B.sh


# export SLIME_ROOT=/mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime
# export MEGATRON_ROOT=/mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/Megatron-LM

# PYTHONPATH=${SLIME_ROOT}:${MEGATRON_ROOT} python ${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py \
#   ${MODEL_ARGS[@]} \
#   --hf-checkpoint /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--yuchenFan--Qwen3-Next-2B-A0.5B-1121/snapshots/eac526354249772cf0c03c5879d908cef1f20706 \
#   --save /mnt/shared-storage-user/p1-shared/luotianwei/checkpoints/Qwen3-Next-2B-A0.5B-Base-1121


export SLIME_ROOT=/mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime
export MEGATRON_ROOT=/mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/Megatron-LM/

source /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/scripts/models/qwen3-next-2B-A0.5B.sh

PYTHONPATH=${SLIME_ROOT}:${MEGATRON_ROOT} python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--yuchenFan--Qwen3-Next-2B-A0.5B-1121/snapshots/eac526354249772cf0c03c5879d908cef1f20706 \
    --save  /mnt/shared-storage-user/p1-shared/luotianwei/checkpoints/Qwen3-Next-2B-A0.5B-Base-1121 \


# source /apdcephfs/mnt/cephfs/users/yuchenfan/slime/scripts/models/qwen3-next-2B-A0.5B.sh

# PYTHONPATH=/apdcephfs/mnt/cephfs/users/yuchenfan/Megatron-LM python tools/convert_hf_to_torch_dist.py \
#     ${MODEL_ARGS[@]} \
#     --hf-checkpoint /apdcephfs/mnt/cephfs/users/yuchenfan/qwen-3-next-2B-A0.5B-baseline-mtp/iter_0008191_hf \
#     --save  /apdcephfs/mnt/cephfs/users/yuchenfan/qwen-3-next-mixed/iter_008192_torch_dist