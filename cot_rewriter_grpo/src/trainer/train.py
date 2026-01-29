#!/usr/bin/env python3
"""
CoT Rewriter GRPO Training Entry Point

This script serves as the entry point for training the CoT Rewriter model
using GRPO (Group Relative Policy Optimization) algorithm.

Usage:
    python src/trainer/train.py
    python src/trainer/train.py --config-name=qwen3_8b_grpo_4gpu
    python src/trainer/train.py trainer.total_epochs=5 data.train_batch_size=32

Key Points:
    - Uses Hydra for configuration management
    - Calls verl.trainer.main_ppo.run_ppo() directly (NOT main())
    - main() in verl is decorated with @hydra.main, so we can't call it with config
"""

import os
import sys
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Add verl to path
VERL_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "verl")
if os.path.exists(VERL_ROOT):
    sys.path.insert(0, VERL_ROOT)

import hydra
from omegaconf import DictConfig, OmegaConf


def setup_environment():
    """Configure environment variables for training."""
    # Ray configuration
    os.environ.setdefault("RAY_DEDUP_LOGS", "1")
    os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")

    # NCCL configuration
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    # Tokenizers parallelism
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Default CUDA devices if not set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def add_timestamp_to_output_dir(config: DictConfig) -> DictConfig:
    """
    Add timestamp to output directory to avoid overwriting.

    This ensures each training run gets a unique directory:
    outputs/project_name/experiment_name/version_YYYYMMDD_HHMMSS/
    """
    if hasattr(config, "trainer") and hasattr(config.trainer, "default_local_dir"):
        # Check if timestamp is already in the path
        if "version_" not in config.trainer.default_local_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = config.trainer.default_local_dir
            config.trainer.default_local_dir = f"{base_dir}/version_{timestamp}"

    return config


def validate_config_local(config: DictConfig) -> None:
    """Validate configuration before training."""
    # Check model paths
    model_path = config.actor_rollout_ref.model.path
    if not os.path.exists(model_path):
        print(f"Warning: Actor model not found: {model_path}")
        print("Please download the model or update the path in config.")

    # Check reward model paths
    custom_reward_fn = config.get("custom_reward_function")
    if custom_reward_fn and custom_reward_fn.get("reward_kwargs"):
        reward_kwargs = custom_reward_fn.reward_kwargs
        if reward_kwargs.get("use_logic_reward", True):
            fsf_model_path = reward_kwargs.get("fsf_model_path")
            if fsf_model_path and not os.path.exists(fsf_model_path):
                print(f"Warning: FSF judge model not found: {fsf_model_path}")
                print("Please download the model or update the path in config.")

        student_model_path = reward_kwargs.get("student_model_path")
        if student_model_path and not os.path.exists(student_model_path):
            print(f"Warning: Student model not found: {student_model_path}")
            print("Please download the model or update the path in config.")

    # Check data paths
    train_files = config.data.train_files
    if not os.path.exists(train_files):
        print(f"Warning: Training data not found: {train_files}")
        print("Please prepare data using: python src/data/dataset.py")

    # Validate GRPO settings
    if config.algorithm.adv_estimator != "grpo":
        print(f"Warning: adv_estimator is '{config.algorithm.adv_estimator}', expected 'grpo'")

    # Validate critic is disabled for GRPO
    if hasattr(config, "critic") and config.critic.get("enable", False):
        print("Warning: Critic is enabled but GRPO doesn't require a critic. Consider disabling.")


def print_config_summary(config: DictConfig) -> None:
    """Print a summary of the training configuration."""
    print("=" * 60)
    print("CoT Rewriter GRPO Training Configuration")
    print("=" * 60)
    print(f"Project: {config.trainer.project_name}")
    print(f"Experiment: {config.trainer.experiment_name}")
    print(f"Output Dir: {config.trainer.default_local_dir}")
    print("-" * 60)
    print(f"Model: {config.actor_rollout_ref.model.path}")
    lora_config = config.actor_rollout_ref.model.get("lora", None)
    if lora_config:
        print(
            f"LoRA: rank={lora_config.get('rank', 'N/A')}, alpha={lora_config.get('alpha', 'N/A')}"
        )
    else:
        print("LoRA: disabled (full parameter training)")
    print(f"Train Batch Size: {config.data.train_batch_size}")
    print(f"GRPO Samples (n): {config.actor_rollout_ref.rollout.n}")
    print(f"Total Epochs: {config.trainer.total_epochs}")
    print("-" * 60)
    print(f"GPUs per Node: {config.trainer.n_gpus_per_node}")
    print(f"Nodes: {config.trainer.nnodes}")
    print("=" * 60)


@hydra.main(
    config_path="../../examples",
    config_name="qwen3_8b_grpo_4gpu",
    version_base=None,
)
def main(config: DictConfig):
    """
    Main training function.

    This function:
    1. Sets up environment variables
    2. Adds timestamp to output directory
    3. Validates configuration
    4. Calls verl's run_ppo() function to start training

    IMPORTANT: We call run_ppo() instead of main() because main() in verl
    is decorated with @hydra.main and expects to be called from command line.

    Args:
        config: Hydra configuration loaded from examples/*.yaml
    """
    setup_environment()

    # Add timestamp to output directory to avoid overwriting previous runs
    config = add_timestamp_to_output_dir(config)

    # Print configuration summary
    print_config_summary(config)

    # Validate configuration (local checks)
    validate_config_local(config)

    # Print full configuration for debugging
    if os.environ.get("DEBUG_CONFIG", "0") == "1":
        print("\nFull Configuration:")
        print(OmegaConf.to_yaml(config))

    # Import and run VERL trainer
    # IMPORTANT: Use run_ppo() instead of main()
    # main() is decorated with @hydra.main and can't be called with config directly
    from verl.trainer.main_ppo import run_ppo

    print("\nStarting VERL GRPO training...")
    print("Note: Ray will be initialized by run_ppo() internally")
    print("-" * 60)

    try:
        # run_ppo handles Ray initialization internally
        run_ppo(config)
        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
