# 任务
我运行训练任务bash scripts/grpo/start_ruc_4gpu_80g.sh的时候报错，请你帮我修复
要求：
1. 你不可以修改数据配置，只能用目前提供好的数据格式
2. 所有模型都必须运行在GPU上
# 报错信息如下
```
[36m(TaskRunner pid=58962)[0m [32m2026-02-02 21:10:46[0m | [1mINFO    [0m | [36mstudent_reward.py:367[0m | [1m[VLLMStudentReward] Waiting for 4 rewrite tasks...[0m
[36m(VLLMWorkerActor pid=62113)[0m Exception in thread EngineCoreOutputQueueThread:
[36m(VLLMWorkerActor pid=62113)[0m Traceback (most recent call last):
[36m(VLLMWorkerActor pid=62113)[0m   File "/home/u20240075/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/threading.py", line 1075, in _bootstrap_inner
[36m(VLLMWorkerActor pid=62113)[0m     self.run()
[36m(VLLMWorkerActor pid=62113)[0m   File "/home/u20240075/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/threading.py", line 1012, in run
[36m(VLLMWorkerActor pid=62113)[0m     self._target(*self._args, **self._kwargs)
[36m(VLLMWorkerActor pid=62113)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 476, in process_outputs_socket
[36m(VLLMWorkerActor pid=62113)[0m     outputs = decoder.decode(frame.buffer)
[36m(VLLMWorkerActor pid=62113)[0m               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(VLLMWorkerActor pid=62113)[0m   File "/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/vllm/v1/serial_utils.py", line 37, in decode
[36m(VLLMWorkerActor pid=62113)[0m     return self.decoder.decode(obj)
[36m(VLLMWorkerActor pid=62113)[0m            ^^^^^^^^^^^^^^^^^^^^^^^^
[36m(VLLMWorkerActor pid=62113)[0m msgspec.ValidationError: Expected `float`, got `str` - at `$[3]`
[36m(VLLMWorkerActor pid=62114)[0m 
```
