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
# 问题
目前fsf的逻辑是fsf v2
但是我希望直接覆盖fsf原来的逻辑，不用在参数或者类名中增加很多fsf_v2的字眼
/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/pandas/core/computation/expressions.py:22: UserWarning: Pandas requires version '2.10.2' or newer of 'numexpr' (version '2.8.7' currently installed).
  from pandas.core.computation.check import NUMEXPR_INSTALLED
/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56: UserWarning: Pandas requires version '1.4.2' or newer of 'bottleneck' (version '1.3.7' currently installed).
  from pandas.core import (
/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Traceback (most recent call last):
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/verl/verl/trainer/main_ppo.py", line 32, in <module>
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, RayDAPOTrainer
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/verl/verl/trainer/ppo/ray_trainer.py", line 64, in <module>
    from verl.experimental.reward.cot_rewriter.training_logger import get_logger as get_training_logger
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/verl/verl/experimental/reward/__init__.py", line 16, in <module>
    from .cot_rewriter import (
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/verl/verl/experimental/reward/cot_rewriter/__init__.py", line 3, in <module>
    from verl.experimental.reward.cot_rewriter.combined_reward import (
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/verl/verl/experimental/reward/cot_rewriter/combined_reward.py", line 248
    fsf_model_path: Optional[str] = None,  # Path to FSF V2 model
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SyntaxError: duplicate argument 'fsf_model_path' in function definition
