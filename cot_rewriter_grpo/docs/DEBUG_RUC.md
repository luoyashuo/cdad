# 任务
我运行训练任务bash scripts/grpo/start_ruc_4gpu_80g.sh, 到第20步OOM，请你帮我检查配置，咋么样降低风险
要求：
1. 你不可以修改数据配置，只能用目前提供好的数据格式
2. 所有模型都必须运行在GPU上
# 报错信息如下
```
[36m(VLLMWorkerActor pid=56794)[0m [VLLMWorker-3] Done: 1 prompts in 0.06s (17.6 samples/s)
[36m(VLLMWorkerActor pid=56794)[0m [VLLMWorker-3] vLLM initialized successfully[32m [repeated 3x across cluster][0m
[36m(VLLMWorkerActor pid=56794)[0m [VLLMWorker-3] nvidia-smi AFTER loading Student vLLM:[32m [repeated 3x across cluster][0m
[36m(VLLMWorkerActor pid=56796)[0m [VLLMWorker-0] GPU 0: 36.0GB / 80.0GB (36823MB used)[32m [repeated 3x across cluster][0m
[36m(VLLMWorkerActor pid=56796)[0m [VLLMWorker-0] ========================================[32m [repeated 3x across cluster][0m
[36m(VLLMWorkerActor pid=56795)[0m [VLLMWorker-2] Processing 16 prompts...[32m [repeated 11x across cluster][0m
[36m(VLLMWorkerActor pid=56795)[0m [VLLMWorker-2] GPU memory BEFORE generate: GPU 2: 71.2GB / 80.0GB (72897MB used)[32m [repeated 11x across cluster][0m
[36m(VLLMWorkerActor pid=56793)[0m [VLLMWorker-1] Done: 16 prompts in 3.61s (4.4 samples/s)[32m [repeated 8x across cluster][0m
[36m(TaskRunner pid=10702)[0m Error in reward_fn: [36mray::VLLMWorkerActor.compute_logprobs()[39m (pid=56794, ip=10.0.0.1, actor_id=66cc5c69256aea184653fa0e01000000, repr=<verl.experimental.reward.cot_rewriter.student_reward.VLLMWorkerActor object at 0x1542447318b0>)
[36m(TaskRunner pid=10702)[0m            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/verl/verl/experimental/reward/cot_rewriter/student_reward.py", line 140, in compute_logprobs
[36m(TaskRunner pid=10702)[0m     outputs = self.llm.generate(prompts, sampling_params)
[36m(TaskRunner pid=10702)[0m               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/vllm/utils.py", line 1134, in inner
[36m(TaskRunner pid=10702)[0m     return fn(*args, **kwargs)
[36m(TaskRunner pid=10702)[0m            ^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/vllm/entrypoints/llm.py", line 470, in generate
[36m(TaskRunner pid=10702)[0m     outputs = self._run_engine(use_tqdm=use_tqdm)
[36m(TaskRunner pid=10702)[0m               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/vllm/entrypoints/llm.py", line 1409, in _run_engine
[36m(TaskRunner pid=10702)[0m     step_outputs = self.llm_engine.step()
[36m(TaskRunner pid=10702)[0m                    ^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/vllm/engine/llm_engine.py", line 1431, in step
[36m(TaskRunner pid=10702)[0m     outputs = self.model_executor.execute_model(
[36m(TaskRunner pid=10702)[0m               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/vllm/executor/executor_base.py", line 140, in execute_model
[36m(TaskRunner pid=10702)[0m     output = self.collective_rpc("execute_model",
[36m(TaskRunner pid=10702)[0m              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/vllm/executor/uniproc_executor.py", line 56, in collective_rpc
[36m(TaskRunner pid=10702)[0m     answer = run_method(self.driver_worker, method, args, kwargs)
[36m(TaskRunner pid=10702)[0m              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/vllm/utils.py", line 2378, in run_method
[36m(TaskRunner pid=10702)[0m     return func(*args, **kwargs)
[36m(TaskRunner pid=10702)[0m            ^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/vllm/worker/worker_base.py", line 420, in execute_model
[36m(TaskRunner pid=10702)[0m     output = self.model_runner.execute_model(
[36m(TaskRunner pid=10702)[0m              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
[36m(TaskRunner pid=10702)[0m     return func(*args, **kwargs)
[36m(TaskRunner pid=10702)[0m            ^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/vllm/worker/model_runner.py", line 1826, in execute_model
[36m(TaskRunner pid=10702)[0m     output: SamplerOutput = self.model.sample(
[36m(TaskRunner pid=10702)[0m                             ^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3.py", line 319, in sample
[36m(TaskRunner pid=10702)[0m     next_tokens = self.sampler(logits, sampling_metadata)
[36m(TaskRunner pid=10702)[0m                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
[36m(TaskRunner pid=10702)[0m     return self._call_impl(*args, **kwargs)
[36m(TaskRunner pid=10702)[0m            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
[36m(TaskRunner pid=10702)[0m     return forward_call(*args, **kwargs)
[36m(TaskRunner pid=10702)[0m            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/sampler.py", line 314, in forward
[36m(TaskRunner pid=10702)[0m     prompt_logprobs, sample_logprobs = get_logprobs(
[36m(TaskRunner pid=10702)[0m                                        ^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/sampler.py", line 903, in get_logprobs
[36m(TaskRunner pid=10702)[0m     ranks = _get_ranks(
[36m(TaskRunner pid=10702)[0m             ^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/sampler.py", line 803, in _get_ranks
[36m(TaskRunner pid=10702)[0m     return result.sum(1).add_(1)
[36m(TaskRunner pid=10702)[0m            ^^^^^^^^^^^^^
[36m(TaskRunner pid=10702)[0m torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 9.00 GiB. GPU 0 has a total capacity of 79.32 GiB of which 5.97 GiB is free. Process 10743 has 13.06 GiB memory in use. Process 15442 has 60.26 GiB memory in use. Of the allocated memory 43.70 GiB is allocated by PyTorch, and 16.06 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)[36m(TaskRunner pid=10702)[0m [32m2026-02-03 16:04:39[0m | [1mINFO    [0m | [36mcombined_reward.py:378[0m | [1m[REWARD_FN] Decoded 64 prompts and 64 responses[0m
[36m(TaskRunner pid=10702)
```
