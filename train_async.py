import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.tracking_utils import init_tracking

# === 修复开始：注册自定义 Config 和 Tokenizer ===
import sys
from transformers import AutoConfig, AutoTokenizer, Qwen2Config, Qwen3NextConfig

# 尝试导入 Qwen2 的 Tokenizer，如果版本太旧可能需要 fallback

from transformers import Qwen2Tokenizer, Qwen2TokenizerFast


class Qwen3KimiConfig(Qwen3NextConfig):
    model_type = "qwen3_kimi"

    def __init__(
        self,
        linear_conv_kernel_dim=4,
        linear_num_value_heads=None,
        linear_num_key_heads=None,
        linear_key_head_dim=None,
        linear_value_head_dim=None,
        attention_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.attention_bias = attention_bias

# 1. 注册 Config

AutoConfig.register("qwen3_kimi", Qwen3KimiConfig)
print("[Slime] Successfully registered 'qwen3_kimi' config.")


AutoTokenizer.register(Qwen3KimiConfig, slow_tokenizer_class=Qwen2Tokenizer, fast_tokenizer_class=Qwen2TokenizerFast)
print("[Slime] Successfully registered 'qwen3_kimi' tokenizer mapping.")
# === 修复结束 ===

# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            actor_model.save_model(rollout_id)
            if args.use_critic:
                critic_model.save_model(rollout_id)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            actor_model.update_weights()

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
