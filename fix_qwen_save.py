#!/usr/bin/env python3
"""
Wrapper script to fix Qwen shared tensor issue during RM training.
Usage: python fix_qwen_save.py llamafactory-cli train examples/fsf_judge/qwen3_1_7b_reward.yaml
"""

import sys
import torch
from unittest.mock import patch

# Patch safetensors.torch.save_file to handle shared tensors
def patched_save_file(tensors, filename, metadata=None):
    from safetensors.torch import save_file as original_save_file
    
    # Detach shared tensors by cloning them
    detached_tensors = {}
    for key, tensor in tensors.items():
        if tensor is not None:
            # Clone to break memory sharing
            detached_tensors[key] = tensor.clone().detach()
        else:
            detached_tensors[key] = tensor
    
    return original_save_file(detached_tensors, filename, metadata=metadata)

# Apply patch before importing llamafactory
with patch('safetensors.torch.save_file', patched_save_file):
    from llamafactory.cli import main
    
if __name__ == "__main__":
    # Remove the script name from args
    sys.argv = sys.argv[1:]
    main()
