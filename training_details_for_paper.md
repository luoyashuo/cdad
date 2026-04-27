# Training Details

## Hardware & Environment

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA A100-PCIE-40GB × 8 |
| CPU | Intel Xeon Processor (Skylake) |
| System Memory | ~630 GB |
| Training Framework | LlamaFactory (SFT) / VERL (GRPO) |

---

## 1. Knowledge Distillation (SFT)

**Entry:** `LlamaFactory/examples/run_exps.sh`

| Hyper-parameter | Full FT (3B) | LoRA (7B) |
|----------------|--------------|-----------|
| Epochs | 2 | 2 |
| Learning Rate | 2.0e-5 | 2.0e-4 |
| LR Scheduler | Cosine | Cosine |
| Warmup Ratio | 0.1 | 0.1 |
| Batch Size | 16 (8 GPUs × 1 × 2) | 16 (8 GPUs × 1 × 2) |
| Max Seq Length | 9,000 | 9,000 |
| Precision | bf16 | bf16 |
| Parallelism | DeepSpeed ZeRO-2 | FlashAttn2 |
| LoRA Rank | — | 128 |
| LoRA Alpha | — | 256 |
| LoRA Target | — | all |
| LoRA Dropout | — | 0.05 |

---

## 2. Reinforcement Learning (GRPO)

**Entry:** `cot_rewriter_grpo/scripts/grpo/start_a100.sh`

### 2.1 Training Config

| Hyper-parameter | Value |
|----------------|-------|
| Algorithm | GRPO |
| Training Steps | 500 |
| Train Batch Size | 16 |
| Group Size | 8 |
| PPO Epochs | 1 |
| Clip Ratio | 0.2 (0.2–0.28) |
| Entropy Coeff | 0.0 |
| Gradient Clipping | 1.0 |

### 2.2 Actor Optimization

| Hyper-parameter | Value |
|----------------|-------|
| LoRA Rank | 32 |
| LoRA Alpha | 64 |
| LoRA Dropout | 0.05 |
| Target Modules | all-linear |
| Learning Rate | 3.0e-5 |
| Weight Decay | 0.01 |
| LR Scheduler | Cosine |
| Warmup Steps Ratio | 0.1 |
| Min LR Ratio | 0.1 |

### 2.3 KL Regularization

| Hyper-parameter | Value |
|----------------|-------|
| KL Loss | true |
| KL Loss Coef | 0.01 |
| KL Loss Type | low_var_kl |
| KL in Reward | false |

### 2.4 Rollout Generation

| Hyper-parameter | Value |
|----------------|-------|
| Engine | vLLM |
| Tensor Parallel | 2 |
| GPU Memory Utilization | 0.6 |
| Temperature | 1.0 (Llama) / 0.7 (Qwen) |
| Top-k | 50 |
| Top-p | 0.95 |
| Presence Penalty | 1.5 |
| Max Prompt Length | 4,800 |
| Max Response Length | 8,192 |

---

## 3. Reward Function

**File:** `verl/verl/experimental/reward/cot_rewriter/combined_reward.py`

### 3.1 Cognitive Reward

| Hyper-parameter | Value | Description |
|----------------|-------|-------------|
| reward_mode | kl | KL-based group-wise reward |
| w_kl_cot | 0.75 | Weight for CoT-part KL divergence |
| w_kl_answer | 0.25 | Weight for Answer-part KL divergence |
| alpha_kl | 0.01 | Weight for max KL term in aggregation |
| kl_eps | 1e-6 | Epsilon for numerical stability |

The cognitive reward is computed via the k3 estimator with group-wise adaptive normalization, using the KL divergence between the actor and the student model on both CoT and answer tokens.

### 3.2 FSF Penalty

FSF Judge detects reasoning-graph failures. If the rewrite has fewer failed nodes than the original, the total reward is set to zero.

| Hyper-parameter | Value |
|----------------|-------|
| use_fsf_penalty | true |
| Model | Yaseal-qwen3_4b_fsf_grm |
| Backend | SGLang |
| Tensor Parallel | 1 |
| Temperature | 0.6 (Llama) / 0.3 (Qwen) |
| Top-k | 20 |
| Top-p | 0.95 |
| Max New Tokens | 3,072 |
| GPU Memory Utilization | 0.35 |
| Max Model Length | 10,000 |
| Micro Batch Size | 1 |

### 3.3 Correctness Judge

| Hyper-parameter | Value |
|----------------|-------|
| use_correct_judge | true |
| Model | Qwen3-8B (Llama) / Qwen3-4B (Qwen) |
| GPU Memory Utilization | 0.50 |
| Max Model Length | 10,000 |
| Max New Tokens | 10 |

Samples judged as incorrect receive zero reward.

### 3.4 Format & Length Penalties

All penalties are binary: if triggered, the sample reward becomes zero.

| Penalty | Llama3.1-8B | Qwen2.5-7B |
|---------|-------------|------------|
| Format | enabled | enabled |
| Truncation | enabled | enabled |
| Too Short | enabled (threshold 0.6) | disabled |
| Too Long | enabled (threshold 2.0) | disabled |
| Repeat | disabled | disabled |

Format rules require the response to be non-empty, contain no problem marker leakage, and not start with rewriting-related keywords.

---

## 4. Data

| Stage | Dataset | Samples |
|-------|---------|---------|
| SFT | DeepMath-Plain-30k | ~30,000 |
| RL | CoT-Rewrite-Mix-5k | ~5,000 |
