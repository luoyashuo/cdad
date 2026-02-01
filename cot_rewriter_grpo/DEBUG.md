# 任务
我运行训练任务bash scripts/grpo/start.sh的时候报错，请你帮我修复
要求：
1. 你不可以修改数据配置，只能用目前提供好的数据格式
2. 所有模型都必须运行在GPU上
# 报错信息如下
```
Error executing job with overrides: []
Traceback (most recent call last):
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/verl/verl/trainer/main_ppo.py", line 42, in main
    run_ppo(config)
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/verl/verl/trainer/main_ppo.py", line 115, in run_ppo
    ray.get(runner.run.remote(config))
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/ray/_private/worker.py", line 2967, in get
    values, debugger_breakpoint = worker.get_objects(
                                  ^^^^^^^^^^^^^^^^^^^
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/ray/_private/worker.py", line 1015, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(RuntimeError): [36mray::TaskRunner.run()[39m (pid=18582, ip=10.0.0.1, actor_id=92a165d853bba70919a6ef0801000000, repr=<main_ppo.TaskRunner object at 0x1477b812dfd0>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/verl/verl/trainer/main_ppo.py", line 368, in run
    trainer.fit()
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/verl/verl/trainer/ppo/ray_trainer.py", line 1279, in fit
    actor_output = self.actor_rollout_wg.update_actor(batch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/verl/verl/single_controller/ray/base.py", line 48, in __call__
    output = ray.get(output)
             ^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^
                                  ^^^^^^^^^^^^^^^^^^^
ray.exceptions.RayTaskError(RuntimeError): [36mray::WorkerDict.actor_rollout_update_actor()[39m (pid=19298, ip=10.0.0.1, actor_id=d228559ba3ad669a817636ab01000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x14b5193a0200>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/verl/verl/single_controller/ray/base.py", line 700, in func
    return getattr(self.worker_dict[key], name)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/verl/verl/single_controller/base/decorator.py", line 442, in inner
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/verl/verl/utils/transferqueue_utils.py", line 199, in dummy_inner
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/verl/verl/utils/profiler/profile.py", line 256, in wrapper
    return func(self_instance, *args, **kwargs_inner)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/verl/verl/workers/fsdp_workers.py", line 877, in update_actor
    metrics = self.actor.update_policy(data=data)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/verl/verl/utils/profiler/performance.py", line 105, in f
    return self.log(decorated_function, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/verl/verl/utils/profiler/performance.py", line 118, in log
    output = func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/verl/verl/workers/actor/dp_actor.py", line 520, in update_policy
    loss.backward()
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/torch/_tensor.py", line 626, in backward
    torch.autograd.backward(
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/torch/autograd/__init__.py", line 347, in backward
    _engine_run_backward(
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/torch/autograd/graph.py", line 823, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/torch/distributed/fsdp/_runtime_utils.py", line 754, in _post_backward_hook
    flat_param.grad.data = flat_param.grad.to(handle._reduce_dtype)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: CUDA driver error: invalid argument

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

```
