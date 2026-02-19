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
/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/pandas/core/computation/expressions.py:22: UserWarning: Pandas requires version '2.10.2' or newer of 'numexpr' (version '2.8.7' currently installed).
  from pandas.core.computation.check import NUMEXPR_INSTALLED
/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56: UserWarning: Pandas requires version '1.4.2' or newer of 'bottleneck' (version '1.3.7' currently installed).
  from pandas.core import (
/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
INFO 02-19 07:58:52 [importing.py:53] Triton module has been replaced with a placeholder.
INFO 02-19 07:58:53 [__init__.py:239] Automatically detected platform cuda.
Traceback (most recent call last):
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
    return func()
           ^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
    lambda: hydra.run(
            ^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/hydra/_internal/hydra.py", line 105, in run
    cfg = self.compose_config(
          ^^^^^^^^^^^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/hydra/_internal/hydra.py", line 594, in compose_config
    cfg = self.config_loader.load_configuration(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/hydra/_internal/config_loader_impl.py", line 142, in load_configuration
    return self._load_configuration_impl(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/hydra/_internal/config_loader_impl.py", line 244, in _load_configuration_impl
    parsed_overrides, caching_repo = self._parse_overrides_and_create_caching_repo(
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/hydra/_internal/config_loader_impl.py", line 230, in _parse_overrides_and_create_caching_repo
    self._process_config_searchpath(config_name, parsed_overrides, caching_repo)
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/hydra/_internal/config_loader_impl.py", line 159, in _process_config_searchpath
    loaded = repo.load_config(config_path=config_name)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/hydra/_internal/config_repository.py", line 348, in load_config
    ret = self.delegate.load_config(config_path=config_path)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/hydra/_internal/config_repository.py", line 91, in load_config
    ret = source.load_config(config_path=config_path)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/hydra/_internal/core_plugins/file_config_source.py", line 31, in load_config
    cfg = OmegaConf.load(f)
          ^^^^^^^^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/omegaconf/omegaconf.py", line 192, in load
    obj = yaml.load(file_, Loader=get_yaml_loader())
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/yaml/__init__.py", line 81, in load
    return loader.get_single_data()
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/yaml/constructor.py", line 51, in get_single_data
    return self.construct_document(node)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/yaml/constructor.py", line 60, in construct_document
    for dummy in generator:
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/yaml/constructor.py", line 413, in construct_yaml_map
    value = self.construct_mapping(node)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/luoys/projects/cdad/cot_rewriter_grpo/.venv/lib/python3.12/site-packages/omegaconf/_utils.py", line 144, in construct_mapping
    raise yaml.constructor.ConstructorError(
yaml.constructor.ConstructorError: while constructing a mapping
  in "/data/luoys/projects/cdad/cot_rewriter_grpo/examples/llama3_3b_grpo_8x80g.yaml", line 100, column 5
found duplicate key enforce_eager
  in "/data/luoys/projects/cdad/cot_rewriter_grpo/examples/llama3_3b_grpo_8x80g.yaml", line 117, column 5
