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
5. 参考原版verl逻辑：/data/luoys/projects/verl-0.5.0
# LOG
```
/data/luoys/projects/cdad/cot_rewriter_grpo/logs/grpo/qwen3_4b_dapo_8x80g_vllm_rollout_sglang_reward.log
```
# 问题
[36m(TaskRunner pid=2014591)[0m https://swanlab.cn/@Yaseal/cot_rewriter_dapo/runs/qgza08tgbcwsuliyufchs
[36m(SGLangWorkerActor pid=2025109)[0m NCCL version 2.21.5+cuda12.4[32m [repeated 3x across cluster][0m
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
ray.exceptions.RayTaskError(TypeError): [36mray::TaskRunner.run()[39m (pid=2014591, ip=10.254.0.16, actor_id=5e4c871e692029944cecefd201000000, repr=<main_ppo.TaskRunner object at 0x146fcc3015b0>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/verl/verl/trainer/main_ppo.py", line 296, in run
    trainer.fit()
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/verl/verl/trainer/ppo/ray_trainer.py", line 1855, in fit
    new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
TypeError: Got unsupported ScalarType BFloat16
