# 任务
训练任务: bash scripts/grpo/start_qc.sh
运行环境(uv)：
```
source /data/luoys/projects/cdad/cot_rewriter_grpo/.venv/bin/activate
```
1. 你不可以修改数据配置，只能用目前提供好的数据格式
2. 所有模型都必须运行在GPU上
3. 运行配置是8*A100-80G
4. 你不能修改数据的长度， response长度
# LOG
/data/luoys/projects/cdad/cot_rewriter_grpo/logs/grpo/llama3_8b_grpo_8x80g.log
# 问题
[36m(TaskRunner pid=2082609)[0m swanlab: 🏠 View project at https://swanlab.cn/@user/cot_rewriter_grpo
[36m(TaskRunner pid=2082609)[0m swanlab: 🚀 View run at https://swanlab.cn/@user/cot_rewriter_grpo/runs/x9iey3nezpf26gt41w7j2
[36m(TaskRunner pid=2082609)[0m Checkpoint tracker file does not exist: /data/luoys/checkpoints/cdad/cot_rewriter_grpo/llama3_3b_grpo_8x80g_top2k_v3/latest_checkpointed_iteration.txt
[36m(TaskRunner pid=2082609)[0m Training from scratch
[36m(WorkerDict pid=2083509)[0m kwargs: {'n': 1, 'logprobs': 0, 'max_tokens': 10000, 'detokenize': False, 'temperature': 0.7, 'top_k': 50, 'top_p': 0.95, 'ignore_eos': False, 'presence_penalty': 1.5}[32m [repeated 7x across cluster][0m
[36m(SGLangWorkerActor pid=2086225)[0m NCCL version 2.21.5+cuda12.4
[36m(SGLangWorkerActor pid=2086224)[0m NCCL version 2.21.5+cuda12.4
[36m(SGLangCorrectJudgeActor pid=2088238)[0m NCCL version 2.21.5+cuda12.4[32m [repeated 3x across cluster][0m
[36m(TaskRunner pid=2082609)[0m swanlab: 🏠 View project at https://swanlab.cn/@Yaseal/cot_rewriter_grpo
[36m(TaskRunner pid=2082609)[0m swanlab: 🚀 View run at 
[36m(TaskRunner pid=2082609)[0m https://swanlab.cn/@Yaseal/cot_rewriter_grpo/runs/x9iey3nezpf26gt41w7j2
[36m(SGLangCorrectJudgeActor pid=2088241)[0m NCCL version 2.21.5+cuda12.4[32m [repeated 3x across cluster][0m
Error executing job with overrides: []
Traceback (most recent call last):
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/verl/verl/trainer/main_ppo.py", line 45, in main
    run_ppo(config)
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/verl/verl/trainer/main_ppo.py", line 100, in run_ppo
    ray.get(runner.run.remote(config))
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/ray/_private/worker.py", line 2967, in get
    values, debugger_breakpoint = worker.get_objects(
                                  ^^^^^^^^^^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/ray/_private/worker.py", line 1015, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(TypeError): [36mray::TaskRunner.run()[39m (pid=2082609, ip=10.254.0.16, actor_id=1f48fbb80d0fe9e47bf80a5301000000, repr=<main_ppo.TaskRunner object at 0x152340563290>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/verl/verl/trainer/main_ppo.py", line 300, in run
    trainer.fit()
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/verl/verl/trainer/ppo/ray_trainer.py", line 1578, in fit
    self._dump_generations(
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/verl/verl/trainer/ppo/ray_trainer.py", line 692, in _dump_generations
    lines.append(json.dumps(entry, ensure_ascii=False))
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/apps/anaconda3/lib/python3.12/json/__init__.py", line 238, in dumps
    **kw).encode(obj)
          ^^^^^^^^^^^
  File "/root/apps/anaconda3/lib/python3.12/json/encoder.py", line 200, in encode
    chunks = self.iterencode(o, _one_shot=True)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/apps/anaconda3/lib/python3.12/json/encoder.py", line 258, in iterencode
    return _iterencode(o, 0)
           ^^^^^^^^^^^^^^^^^
  File "/root/apps/anaconda3/lib/python3.12/json/encoder.py", line 180, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type ndarray is not JSON serializable
