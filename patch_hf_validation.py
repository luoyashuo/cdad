"""
Monkey patch to disable huggingface_hub repo_id validation for local paths
"""
import os
import sys

# Only apply patch if using local model paths
def patch_huggingface_hub():
    try:
        from huggingface_hub.utils import _validators

        # Store original function
        original_validate_repo_id = _validators.validate_repo_id

        def patched_validate_repo_id(repo_id, *args, **kwargs):
            # If it's an absolute path, skip validation
            if os.path.isabs(repo_id) and os.path.exists(repo_id):
                return
            # Otherwise use original validation
            return original_validate_repo_id(repo_id, *args, **kwargs)

        # Apply the patch
        _validators.validate_repo_id = patched_validate_repo_id
        print("[PATCH] Successfully patched huggingface_hub.utils._validators.validate_repo_id")
    except Exception as e:
        print(f"[PATCH] Warning: Could not patch validate_repo_id: {e}")

if __name__ == "__main__":
    patch_huggingface_hub()
