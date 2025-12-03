export SLIME_ROOT=/mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime
export MEGATRON_ROOT=/mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/Megatron-LM/

PYTHONPATH=${SLIME_ROOT}:${MEGATRON_ROOT} python tools/convert_torch_dist_to_hf.py \
  --input-dir /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/checkpoints/qwen-3-next-2B-A0.5B-pretrain-kimi/iter_0000173 \
  --output-dir /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/checkpoints/qwen-3-next-kimi-2B-A0.5B/qwen3-next-kimi-2B-A0.5B-sft-iter_0000173 \
  --origin-hf-dir /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--yuchenFan--Qwen3-Next-Kimi-2B-A0.5B/snapshots/04ce5e7db97ab611adb548ff0da4a243cb0d973c \
  --force