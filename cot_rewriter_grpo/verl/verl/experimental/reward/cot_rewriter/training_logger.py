"""
Training Logger for CoT Rewriter GRPO

Provides structured logging for training process using standard logging library.
Logs are only printed at specific key points:
- Rollout start/end: GPU memory, prompt token stats, sample prompts/responses
- Reward start/end: GPU memory, data stats
- Backward start/end: GPU memory

Usage:
    from verl.experimental.reward.cot_rewriter.training_logger import get_logger
    
    logger = get_logger("trainer")
    logger.log_gpu_memory("[ROLLOUT_START]")
    logger.log_prompt_stats(prompts)
"""

import logging
import os
import random
import subprocess
import sys
from typing import List, Optional, Any

import torch

# Global logger instances
_loggers = {}


def get_logger(name: str = "cot_rewriter") -> "TrainingLogger":
    """Get or create a TrainingLogger instance."""
    if name not in _loggers:
        _loggers[name] = TrainingLogger(name)
    return _loggers[name]


class TrainingLogger:
    """Structured logger for training process."""
    
    def __init__(self, name: str, level: Optional[str] = None):
        """Initialize logger.
        
        Args:
            name: Logger name
            level: Logging level (default from env VERL_LOGGING_LEVEL or INFO)
        """
        self.logger = logging.getLogger(name)
        
        # Set level
        if level is None:
            level = os.getenv("VERL_LOGGING_LEVEL", "INFO")
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []
        
        # Add handler if not already added
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(self.logger.level)
            
            # Format: timestamp | level | name | message
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def _log(self, level: int, msg: str, *args, **kwargs):
        """Internal log method."""
        self.logger.log(level, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def log_gpu_memory(self, prefix: str = ""):
        """Log GPU memory usage for all available GPUs.
        
        Args:
            prefix: Prefix string for the log message (e.g., "[ROLLOUT_START]")
        """
        # Try using PyTorch CUDA API first (more accurate)
        if torch.cuda.is_available():
            try:
                gpu_info = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                    total = props.total_memory / 1024**3  # GB
                    gpu_info.append(
                        f"GPU{i}:{allocated:.1f}G/{total:.1f}G(allocated) "
                        f"reserved:{reserved:.1f}G"
                    )
                self.info(f"{prefix} GPU_MEMORY {' | '.join(gpu_info)}")
                return
            except Exception as e:
                self.warning(f"{prefix} Failed to get GPU memory via PyTorch: {e}, falling back to nvidia-smi")
        
        # Fallback: use nvidia-smi command (works even when torch.cuda.is_available() is False)
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 3:
                    gpu_idx = parts[0].strip()
                    used_mb = int(parts[1].strip())
                    total_mb = int(parts[2].strip())
                    used_gb = used_mb / 1024
                    total_gb = total_mb / 1024
                    gpu_info.append(f"GPU{gpu_idx}:{used_gb:.1f}G/{total_gb:.1f}G")
            
            if gpu_info:
                self.info(f"{prefix} GPU_MEMORY {' | '.join(gpu_info)} (via nvidia-smi)")
            else:
                self.info(f"{prefix} GPU not available")
        except Exception as e:
            self.info(f"{prefix} GPU not available (failed to query: {e})")
    
    def log_prompt_stats(
        self, 
        prompts: List[torch.Tensor], 
        prefix: str = "[ROLLOUT]",
        sample_count: int = 2,
        max_tokens_to_print: int = 200,
        tokenizer: Optional[Any] = None
    ):
        """Log prompt token statistics and sample prompts.
        
        Args:
            prompts: List of prompt token tensors
            prefix: Prefix for log messages
            sample_count: Number of random samples to print
            max_tokens_to_print: Max number of tokens to print per sample
            tokenizer: Tokenizer to decode prompts (optional)
        """
        if not prompts:
            self.info(f"{prefix} No prompts to log")
            return
        
        # Compute token length stats
        lengths = [p.shape[0] if len(p.shape) > 0 else len(p) for p in prompts]
        max_len = max(lengths)
        min_len = min(lengths)
        mean_len = sum(lengths) / len(lengths)
        
        self.info(
            f"{prefix} PROMPT_TOKENS max={max_len} min={min_len} mean={mean_len:.1f} "
            f"count={len(prompts)}"
        )
        
        # Randomly sample and print prompts
        if sample_count > 0 and tokenizer is not None:
            sample_indices = random.sample(
                range(len(prompts)), 
                min(sample_count, len(prompts))
            )
            for idx in sample_indices:
                prompt_tokens = prompts[idx]
                # Take first max_tokens_to_print tokens
                tokens_to_decode = prompt_tokens[:max_tokens_to_print]
                prompt_text = tokenizer.decode(tokens_to_decode, skip_special_tokens=True)
                self.info(f"{prefix} SAMPLE_PROMPT[{idx}]: {prompt_text[:200]}...")
    
    def log_response_samples(
        self,
        responses: List[torch.Tensor],
        prefix: str = "[ROLLOUT]",
        sample_count: int = 2,
        max_tokens_to_print: int = 200,
        tokenizer: Optional[Any] = None
    ):
        """Log random response samples.
        
        Args:
            responses: List of response token tensors
            prefix: Prefix for log messages
            sample_count: Number of random samples to print
            max_tokens_to_print: Max number of tokens to print per sample
            tokenizer: Tokenizer to decode responses
        """
        if not responses or tokenizer is None:
            return
        
        sample_indices = random.sample(
            range(len(responses)),
            min(sample_count, len(responses))
        )
        for idx in sample_indices:
            response_tokens = responses[idx]
            tokens_to_decode = response_tokens[:max_tokens_to_print]
            response_text = tokenizer.decode(tokens_to_decode, skip_special_tokens=True)
            self.info(f"{prefix} SAMPLE_RESPONSE[{idx}]: {response_text[:200]}...")
    
    def log_data_stats(
        self,
        data: Any,
        prefix: str = "[REWARD]",
        tokenizer: Optional[Any] = None,
        field: str = "prompts"
    ):
        """Log data statistics (for reward computation).
        
        Args:
            data: Data object containing tensors
            prefix: Prefix for log messages
            tokenizer: Tokenizer (optional)
            field: Field name to compute stats for
        """
        try:
            if hasattr(data, 'batch') and field in data.batch:
                tensor = data.batch[field]
                # Compute sequence lengths (non-padding tokens)
                if tensor.dim() >= 2:
                    # [batch_size, seq_len] - count non-zero tokens
                    lengths = (tensor != 0).sum(dim=-1).tolist()
                else:
                    lengths = [len(tensor)]
                
                max_len = max(lengths)
                min_len = min(lengths)
                mean_len = sum(lengths) / len(lengths)
                
                self.info(
                    f"{prefix} DATA_TOKENS field={field} max={max_len} min={min_len} "
                    f"mean={mean_len:.1f} count={len(lengths)}"
                )
            else:
                self.debug(f"{prefix} Cannot compute stats for field={field}")
        except Exception as e:
            self.debug(f"{prefix} Failed to compute data stats: {e}")
    
    def log_response_preview(
        self,
        responses: torch.Tensor,
        prefix: str = "[REWARD]",
        tokenizer: Optional[Any] = None,
        num_tokens: int = 10
    ):
        """Log first few tokens of responses (excluding padding).
        
        Args:
            responses: Response token tensor [batch_size, seq_len]
            prefix: Prefix for log messages
            tokenizer: Tokenizer to decode tokens
            num_tokens: Number of tokens to print
        """
        if responses is None or tokenizer is None:
            return
        
        try:
            batch_size, seq_len = responses.shape
            
            # Get pad_token_id from tokenizer, default to 0
            pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
            if pad_token_id is None:
                pad_token_id = 0
            
            self.info(f"{prefix} RESPONSE_PREVIEW shape={responses.shape}, pad_id={pad_token_id}:")
            
            for i in range(min(3, batch_size)):  # Show first 3 samples
                response_ids = responses[i].tolist()
                
                # Find first non-padding token position
                first_non_pad_pos = 0
                while first_non_pad_pos < len(response_ids) and response_ids[first_non_pad_pos] == pad_token_id:
                    first_non_pad_pos += 1
                
                # Find last non-padding token position
                last_non_pad_pos = len(response_ids) - 1
                while last_non_pad_pos >= 0 and response_ids[last_non_pad_pos] == pad_token_id:
                    last_non_pad_pos -= 1
                
                actual_len = max(0, last_non_pad_pos - first_non_pad_pos + 1)
                
                # Extract actual content (skip left padding)
                content_ids = response_ids[first_non_pad_pos:first_non_pad_pos + num_tokens]
                
                if not content_ids:
                    self.info(f"  [{i}] (empty response, all padding)")
                    continue
                
                # Try to decode
                try:
                    text = tokenizer.decode(content_ids, skip_special_tokens=True)
                    # Remove newlines for compact display
                    text_compact = text.replace('\n', '↵').replace('\r', '')[:60]
                    self.info(f"  [{i}] pos={first_non_pad_pos}-{last_non_pad_pos} len={actual_len} ids={content_ids}")
                    self.info(f"  [{i}] text: {text_compact}...")
                except Exception as decode_err:
                    self.info(f"  [{i}] ids={content_ids} (decode error: {decode_err})")
        except Exception as e:
            self.debug(f"{prefix} Failed to log response preview: {e}")


# Convenience function for quick logging
def log_gpu_memory(prefix: str = ""):
    """Quick GPU memory logging."""
    get_logger().log_gpu_memory(prefix)


def log_prompt_stats(prompts: List[torch.Tensor], **kwargs):
    """Quick prompt stats logging."""
    get_logger().log_prompt_stats(prompts, **kwargs)
