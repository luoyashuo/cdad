# Eval V2 设计文档 - 跨Benchmark流式推理与统计汇总

## 1. 核心目标

### 1.1 解决的问题
- **GPU利用率低**：小benchmark无法填满8卡batch，vLLM continuous batching性能浪费
- **串行瓶颈**：测完bench A才能测bench B，GPU在评分阶段空闲
- **固定repeat**：无法为不同benchmark配置不同的repeat次数（如AIME 64次，MATH500 8次）
- **结果分散**：多次运行的指标分散在各文件中，缺乏统一汇总

### 1.2 核心需求
1. **跨benchmark流式推理**：多个bench混合batch，GPU持续满载运行
2. **并行评分**：bench推理完成后立即触发CPU评分，不阻塞GPU继续推理其他bench
3. **差异化repeat**：每个benchmark可独立配置repeat次数
4. **细粒度记录**：分别记录每个bench每次repeat的指标
5. **统一汇总**：所有bench的统计信息（mean/std/min/max）汇总到一个表格

---

## 2. 系统架构

### 2.1 整体流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              主控流程                                    │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │  Bench A     │   │  Bench B     │   │  Bench C     │   ...          │
│  │  (30题×64次) │   │  (500题×8次) │   │  (1000题×1次)│                │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                │
│         │                  │                  │                         │
│         └──────────────────┼──────────────────┘                         │
│                            ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │              Global Prompt Queue (Ray Queue)                  │      │
│  │  所有bench的prompts混合在一起，按提交顺序         │      │
│  │              │      │
│  └──────────────────────────┬────────────────────────────────────┘      │
│                             │                                           │
│  ┌──────────────────────────┼────────────────────────────────────┐     │
│  │                          │                                    │     │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐      │     │
│  │  │Worker 0 │   │Worker 1 │   │Worker 2 │   │Worker 7 │      │     │
│  │  │(GPU 0-1)│   │(GPU 2-3)│   │(GPU 4-5)│   │(GPU 6-7)│      │     │
│  │  │ TP=2   │   │ TP=2   │   │ TP=2   │   │ TP=2   │      │     │
│  │  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘      │     │
│  │       │             │             │             │           │     │
│  │       └─────────────┴─────────────┴─────────────┘           │     │
│  │                          │                                  │     │
│  │  Ray Actor Pool (DP=4, TP=2 each)                          │     │
│  │  持续从Queue取batch，不分bench边界                          │     │
│  └──────────────────────────┼──────────────────────────────────┘     │
│                             │                                         │
│  ┌──────────────────────────▼──────────────────────────────────┐     │
│  │                  Result Queue (Ray Queue)                    │     │
│  │  (prompt_metadata, generated_text, tokens)                   │     │
│  └──────────────────────────┬──────────────────────────────────┘     │
│                             │                                         │
└─────────────────────────────┼─────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Bench-Aware Collector                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  按benchmark分组收集results，追踪每个bench的完成进度              │  │
│  │                                                                  │  │
│  │  AIME25: [████████████████████░░░░░░░░] 1536/1920 (80%) 推理中   │  │
│  │  AMC:    [████████████░░░░░░░░░░░░░░░░] 768/2560 (30%)  推理中   │  │
│  │  MATH500:[████████████████████████░░░░] 3000/4000 (75%) 等待中   │  │
│  │                                                                  │  │
│  │  当某bench达到100% → 触发评分回调（独立线程，不阻塞推理）         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌────────────┐  ┌────────────┐  ┌────────────┐
       │ AIME25     │  │ AMC        │  │ MATH500    │
       │ 评分线程   │  │ 评分线程   │  │ 评分线程   │  CPU ThreadPool
       │ (evalscope)│  │ (evalscope)│  │ (evalscope)│  每个bench独立
       └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
             │               │               │
             └───────────────┼───────────────┘
                             │
                             ▼
              ┌─────────────────────────────────────┐
              │        结果持久化 & 统计汇总         │
              │                                     │
              │  1. 每个bench每次repeat的结果        │
              │     → {bench}_run_{i}_result.json   │
              │                                     │
              │  2. 每个bench的统计信息              │
              │     → {bench}_summary.json          │
              │                                     │
              │  3. 所有bench的总表                  │
              │     → unified_summary.xlsx          │
              │     → unified_summary.json          │
              └─────────────────────────────────────┘
```

---

## 3. 关键流程详解

### 3.1 初始化阶段

**输入**：配置文件（兼容eval.py格式）

```yaml
model_path: /data/cdad/checkpoints/xxx
vllm:
  tp: 1              # Tensor Parallel
  dp: 8              # Data Parallel (Ray Workers数量)
  worker_chunk_size: 128

benchmarks:
  AIME25:
    dataset_name: "aime25"
    repeats: 64      # AIME25运行64次
    temperature: 0.6
    max_tokens: 28000
    
  MATH500:
    dataset_name: "math_500"
    repeats: 8       # MATH500运行8次
    temperature: 0.6
    max_tokens: 28000
    
  GPQA:
    dataset_name: "gpqa_extend"
    repeats: 1       # GPQA运行1次
    temperature: 0.6
    max_tokens: 28000
```

**处理步骤**：
1. 加载所有benchmark的数据集，获取sample数量
2. 为每个sample生成对应repeat次数的prompts
3. 计算总prompts数：`sum(bench.num_samples × bench.repeats)`
4. 初始化Ray集群，启动DP个vLLM Worker

### 3.2 流式推理阶段

**核心机制**：
- 所有bench的prompts进入同一个Global Queue
- 每个Ray Worker独立从Queue取batch进行推理
- 不分bench边界，哪个Worker空闲就处理哪个batch
- 相同prompt（不同repeat）自动去重，使用vLLM的`n`参数共享KV cache

**调度策略**：
- **Batch Size**：每个Worker一次处理`worker_chunk_size`个prompts
- **优先级**：小数据集的prompts优先入队（让它们先完成，尽快释放评分资源）
- **负载均衡**：Ray内置调度，自动平衡各Worker负载

### 3.3 完成检测与评分触发

**Bench完成条件**：
```
某bench的所有prompts都已完成推理
即：completed_count == num_samples × repeats
```

**触发时机**：
- 不等待其他bench
- 该bench推理完成100%立即触发评分
- 评分在独立CPU线程执行，不阻塞GPU推理

**评分流程**：
1. 将该bench的所有results写入临时predictions目录（evalscope格式）
2. 调用evalscope的`run_task()`，设置`use_cache`指向临时目录
3. evalscope执行评分（无推理，纯CPU计算）
4. 保存评分结果

### 3.4 进度显示

**实时展示**：
```
【总进度】███████████████████░░░░░░░░░  8480/12000 [71%] [14:23<05:45, 12.3it/s]

【AIME25     】████████████████████████████░░░░░░  1536/1920 [80%] 推理中
【AMC        】████████████░░░░░░░░░░░░░░░░░░░░░░  768/2560 [30%]  推理中  
【MATH500    】████████████████████████░░░░░░░░░░  3000/4000 [75%] 等待评分
【GPQA       】███████████████████████████████████  448/448 [100%] ✓ 完成
```

**说明**：
- 总进度：所有bench的推理完成进度
- 单bench进度：该bench的prompts完成比例
- 状态标签：推理中 / 评分中 / 等待中 / 完成

---

## 4. 输出格式与统计汇总

### 4.1 文件结构

```
{model_path}/evalscope/{version}/
```
自行完善设计，只需要保证比如对于MATH500,repeat 8;AIME repeat 64; 
那么每次MATH500 8次运行的指标保存到1个json文件，AIME25 64次运行的指标保存到1个json文件
最后把MATH500和AIME25的指标的mean,std,range保存到一个xlsx文件和json文件中




## 6. 与Eval.py的兼容性

### 6.1 配置兼容

完全复用eval.py的配置格式：
/home/luoys/cdad/eval/eval_configs/eval_all_baseline.yaml

### 6.2 输出兼容

- 单repeat结果格式与evalscope原生格式一致
- summary.xlsx格式与eval.py一致
- 新增文件（per_repeat_scores, unified_summary）为扩展，不影响原有流程

### 6.3 启动方式

```bash
python eval_v2.py eval_configs/xxx.yaml
```

---

## 7. 性能预期

### 7.1 GPU利用率对比

| 场景 | 传统流程 | Eval V2 | 提升 |
|------|---------|---------|------|
| AIME(1920) + MATH(4000) | 推理→评分→推理→评分 | 混合推理→并行评分 | 1.5-2x |
| 3个小bench各1000 prompts | 串行，每次预热 | 一次预热，持续满载 | 2-3x |
| 总prompts 10000 | GPU利用率~50% | GPU利用率~90% | 1.8x |

### 7.2 评分并行度

假设4个bench同时完成推理：
- 4个CPU线程并行评分
- 每个线程独立调用evalscope
- 评分速度取决于数据量，互不阻塞

---

## 8. 扩展性设计


### 8 断点续传（Future）

- 记录每个bench的完成状态
- 中断后重新运行，跳过已完成的bench
- 支持bench级别的细粒度续传

---

## 9. 断点续传（Resume）设计

### 9.1 续传粒度

支持两个层面的断点续传：
1. **Inference Level**: 已推理完成的sample无需重新推理
2. **Judge Level**: 已评分完成的bench无需重新评分


## 10. 全局TPS监控

### 10.1 监控目标

- **实时 throughput**: tokens/s, samples/s
- **聚合统计**: 各Worker的贡献，平均batch size
- **历史趋势**: TPS随时间变化曲线
