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
/data/luoys/projects/cdad/cot_rewriter_grpo/logs/grpo/qwen3_4b_gspo_8x80g.log
```
# 问题
帮我实现以下功能，支持projects/cdad/cot_rewriter_grpo/examples/qwen3_4b_gspo_8x80g_v2.yaml这个config
  1. 数据集格式支持升级 (rl_dataset_jsonl.py)

  新增功能：
  - 支持 system_prompt + user_prompt 新格式
  - 兼容 3 种数据格式：
    a. 新格式：system_prompt + user_prompt
    b. Chat 格式：消息列表
    c. 纯文本格式：单字符串
  - 覆盖 maybe_filter_out_long_prompts 方法以正确处理新格式
  - 添加调试日志（首次打印 input_ids decode 结果）

  ---
  2. 奖励计算功能增强 (combined_reward.py, sglang_student_reward.py, student_reward.py)

  新增参数：
  | 参数                    | 类型 | 默认值 | 说明                                                             |
  |-------------------------|------|--------|------------------------------------------------------------------|
  | reward_only_use_logprob | bool | False  | 为 True 时，reward = answer_log_prob(rewrite) only（不计算差值） |
  | use_useful_reward       | bool | True   | 是否使用 R_useful 奖励                                           |

  新增日志：
  - reward_deltas - 记录 rewrite 与 original 的差值（用于监控）