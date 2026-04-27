# Implementation Details

## 1. System Architecture

### 1.1 Hardware Allocation

Our training system runs on a single node with 8× NVIDIA A100-80GB GPUs. The GPUs are partitioned into two functional zones:

| GPU ID | Component | Model | Purpose |
|--------|-----------|-------|---------|
| 0–2 | Actor + Rollout + Training | Qwen3-8B (w/ LoRA) | Policy model that generates rewritten CoT; performs GRPO updates |
| 3 | Student Model | Qwen2.5-7B-Instruct | Computes cognitive reward $R_{\text{cog}}$ via log-probability |
| 3 | FSF Judge | Qwen2.5-7B-Instruct | Converts CoT to reasoning graph (DOT) for GED computation |
| 3 | Correct Judge | Qwen3-4B | Verifies whether rewrite leads to the correct answer |

GPU 0–2 are dedicated to the training pipeline. GPU 3 hosts all reward-computation models, which are loaded **sequentially** (not simultaneously) to fit within a single GPU's memory budget.

### 1.2 Component Responsibilities

**Actor (Policy Model).** Initialized from Qwen3-8B-Instruct with LoRA (rank=32, alpha=64). It receives a prompt consisting of the question and the original CoT, and generates a rewritten CoT. The actor weights are sharded via FSDP and shared with the rollout engine through VERL's hybrid-engine mechanism.

**Rollout Engine (vLLM).** Uses the actor weights to generate $G=8$ responses per question via sampling (temperature=0.7, top-p=0.95). Tensor parallelism=2 is used within GPU 0–2. After rollout, vLLM memory is released (sleep_level=1) before reward models load on GPU 3.

**Reference Model.** A frozen copy of the actor (without LoRA adapters) loaded on GPU 0–2 via FSDP. It provides reference log-probabilities for KL-divergence computation.

**Student Model.** Loaded on GPU 3 via SGLang. It computes the conditional log-probability of the ground-truth answer given $(\text{question}, \text{rewrite\_CoT})$ and given $(\text{question}, \text{original\_CoT})$. The cognitive reward is derived from the difference.

**FSF Judge.** Loaded on GPU 3 via SGLang after the student model is unloaded. It prompts the judge model to convert a CoT into a Graphviz DOT representation of the reasoning graph. The graph edit distance (GED) between the original and rewrite graphs yields the logic reward.

**Correct Judge.** Loaded on GPU 3 via SGLang after the FSF judge is unloaded. A lightweight Qwen3-4B model classifies whether the rewritten CoT leads to the correct solution. If the verdict is "No", the sample receives zero reward and may be excluded from training.

### 1.3 Asynchronous Reward Computation Flow

The reward computation is decoupled from the training loop via **Ray remote actors** to prevent GPU 0–2 from idling:

```
Step 1: Training GPUs (0-2)
  ├─ Actor generates G rollouts per question
  ├─ Reference model computes ref_log_probs
  └─ Send (questions, original_cots, rewrite_cots, answers) to Ray reward worker

Step 2: Reward GPU (3) — Sequential model loading
  ├─ [Load Student Model]
  │   └─ Compute R_cog (logprob / KL divergence)
  │   └─ cleanup() → release GPU memory
  ├─ [Load FSF Judge]      (only if logic reward enabled)
  │   └─ Generate DOT graphs for original & rewrite
  │   └─ Compute GED → R_logic
  │   └─ shutdown() → release GPU memory
  ├─ [Load Correct Judge]  (only if correctness check enabled)
  │   └─ Verify correctness
  │   └─ shutdown() → release GPU memory
  └─ Return reward_tensor + extra_info to training GPUs

Step 3: Training GPUs (0-2)
  ├─ Apply KL penalty to token-level rewards
  ├─ Compute GRPO advantages
  └─ Perform PPO policy update
```

Because reward computation happens asynchronously on GPU 3, the training GPUs can proceed with gradient accumulation or the next data batch while rewards are being computed. The sequential loading strategy on GPU 3 ensures that at most one large model resides in GPU memory at any time, avoiding out-of-memory errors on a single 80 GB GPU.

---

## 2. Model and Training Configuration

### 2.1 Actor (Policy) Model

| Hyper-parameter | Value | Description |
|-----------------|-------|-------------|
| Base model | Qwen2.5-7B-Instruct / Qwen3-8B | Pre-trained instruction-tuned LLM |
| Fine-tuning | LoRA | Parameter-efficient fine-tuning |
| LoRA rank $r$ | 32 | Rank of low-rank adapters |
| LoRA alpha | 64 | Scaling factor; $\alpha/r = 2$ |
| LoRA dropout | 0.05 | Dropout on adapter weights |
| Target modules | `all-linear` | Apply LoRA to all linear layers |
| Gradient checkpointing | true | Trade compute for memory |
| Optimizer | AdamW | — |
| Learning rate | $3 \times 10^{-5}$ | Higher than full fine-tuning due to LoRA |
| LR scheduler | cosine | With 10% warmup steps |
| Min LR ratio | 0.1 | Final LR = $3 \times 10^{-6}$ |
| Weight decay | 0.01 | — |
| Gradient clipping | 1.0 | Max norm for global gradient clipping |
| Mixed precision | bf16 | `param_dtype=bf16`, `reduce_dtype=fp32` |
| FSDP wrap policy | `Qwen2DecoderLayer` | Transformer layer granularity |
| Parameter offload | true | Offload FSDP shards to CPU when unused |
| Optimizer offload | true | Offload optimizer states to CPU |

### 2.2 Rollout (Generation) Configuration

| Hyper-parameter | Value | Description |
|-----------------|-------|-------------|
| Engine | vLLM V0 | Fast LLM inference engine |
| Tensor parallelism | 2 | Shard rollout engine across 2 GPUs |
| Group size $G$ | 8 | Number of responses sampled per prompt |
| Temperature | 0.7 | Sampling temperature |
| Top-p | 0.95 | Nucleus sampling |
| Top-k | 50 | Top-k sampling |
| Presence penalty | 1.5 | Discourage repetition |
| Max prompt length | 4800 tokens | — |
| Max response length | 8192 tokens | — |
| GPU memory utilization | 0.6 | Fraction of GPU memory for vLLM KV cache |
| Sleep level | 1 | Release vLLM memory after rollout (not full offload) |

### 2.3 GRPO Algorithm Configuration

| Hyper-parameter | Value | Description |
|-----------------|-------|-------------|
| Advantage estimator | `grpo` | Group Relative Policy Optimization |
| Normalize by std | true | Divide advantage by group std (original GRPO); set false for Dr.GRPO |
| Clip range low $\varepsilon_{\text{low}}$ | 0.2 | PPO clip lower bound |
| Clip range high $\varepsilon_{\text{high}}$ | 0.28 | PPO clip upper bound |
| Dual-clip ratio $c$ | 3.0 | Lower-bound clip ratio for negative advantages |
| KL loss type | `low_var_kl` | Low-variance KL estimator |
| KL loss coefficient $\beta$ | 0.01 | Weight of KL divergence term in actor loss |
| In-reward KL | false | KL penalty applied as loss term, not subtracted from rewards |
| Entropy coefficient | 0.0 | No entropy bonus |
| PPO epochs per batch | 1 | Single update per rollout batch |
| Loss aggregation | `token-mean` | Average loss over all valid tokens |
| Discount $\gamma$ | 1.0 | No discount for outcome rewards |
| GAE $\lambda$ | 1.0 | Not used in GRPO (placeholder) |

### 2.4 Reward Model Configuration

| Hyper-parameter | Value | Description |
|-----------------|-------|-------------|
| **Student Model** | | |
| Model | Qwen2.5-7B-Instruct | Frozen student for cognitive reward |
| Backend | SGLang | Inference engine |
| Tensor parallelism | 1 | Single-GPU execution |
| Micro batch size | 1 | Avoid OOM with long sequences |
| Max model length | 8192 | — |
| Memory fraction | 0.50 | SGLang static memory allocation |
| **Cognitive Reward** | | |
| Reward mode | `kl` | KL-based reward (alternatives: `logprob`, `ppl`) |
| $\alpha$ (min-log-prob weight) | 0.01 | Weight for worst-token penalty |
| $\alpha_{\text{kl}}$ | 0.01 | Weight for $\max(\text{k3}_t)$ term |
| Use reward delta | true | $R_{\text{cog}} = \text{rewrite} - \text{original}$ |
| **Logic Reward (GED)** | | |
| FSF model | Yaseal-qwen3_4b_fsf_grm | Converts CoT to DOT graph |
| $\lambda_{\text{logic}}$ | 0.5 | Weight of logic reward in total reward |
| GED timeout | 5.0 s | Per-sample NetworkX GED solver timeout |
| GED sigmoid $\alpha$ | 0.5 | Steepness of GED-to-reward mapping |
| Logic reward group size | 8 | Group-wise normalization (matches rollout $G$) |
| Repeat FSF $N$ | 1 | Independent FSF evaluations per sample |
| **Correct Judge** | | |
| Model | Qwen3-4B | Lightweight correctness classifier |
| GPU memory fraction | 0.50 | — |
| Max new tokens | 10 | Short output (`<judge>Yes</judge>` or `<judge>No</judge>`) |
| Filter incorrect samples | true | Exclude "No" samples from GRPO training |
| **Penalties** | | |
| Format penalty | enabled | Zero reward if response format invalid |
| Correctness penalty | enabled | Zero reward if Correct Judge says "No" |
| Too-short penalty | disabled | (Configurable; currently off) |
| Too-long penalty | disabled | (Configurable; currently off) |
| Repeat penalty | disabled | (Configurable; currently off) |

### 2.5 Training Batch Configuration

| Hyper-parameter | Value | Description |
|-----------------|-------|-------------|
| Train batch size | 16 | Number of unique prompts per global batch |
| Effective batch size | 128 | $16 \times G$ (8 responses per prompt) |
| PPO mini-batch size | 16 | Size of each PPO update sub-batch |
| PPO micro-batch size per GPU | 1 | Forward/backward granularity per GPU |
| Gen batch size | 16 | Prompt batch size for rollout generation |
| Validation batch size | 16 | — |
| Dataloader workers | 8 | CPU workers for data loading |
| Total training steps | 500 | — |
| Checkpoint save freq | 10 steps | — |
| Validation freq | 10 steps | — |
