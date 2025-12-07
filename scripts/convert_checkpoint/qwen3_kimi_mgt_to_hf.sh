export SLIME_ROOT=/mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime
export MEGATRON_ROOT=/mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/Megatron-LM/

PYTHONPATH=${SLIME_ROOT}:${MEGATRON_ROOT} python tools/convert_torch_dist_to_hf.py \
  --input-dir /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/checkpoints/qwen-3-next-kimi-1204/iter_0000173 \
  --output-dir /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/hf_checkpoints/qwen3-kimi-1204-iter0000173 \
  --origin-hf-dir /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--yuchenFan--Qwen3-Next-Kimi-1204/snapshots/fccadc7499f8a9bcef45f30f409143e01640bef7 \
  --force