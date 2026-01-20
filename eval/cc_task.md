# 帮我实现一个调用eval scope的测评的脚本
eval scope文档：/home/luoyashuo/research/graph_kd/cdad/eval/evalscope/docs
1. 需要测试的Bench：
AIME 24
AIME 25
GSM8k
MATH 500
AMC 23
GPQA-Diamond
MBPP
HumanEval
2. 脚本输入：
一个yaml文件（例如eval/xxx.yaml）,包括以下内容：
    - model_path: /mnt/moonfs/luoyashuo-ksyun/research/cdad/ckpt/baseline/qwen3_1_7b_base_sft_cg/checkpoint-23944
    - 输入config:
        - 每个benchmark的测评次数
        - 每个benchmark的eval scope的配置
    - vllm: # 配置vllm的启动参数
        - tp: 等vllm基础配置参数
3. 输出：
- 测评结果：保存在{model_path}/evalscope/version_{timestamp}目录下:
    - 一个excel文件为汇总的测试结果（2行，n列，每列为一个benchmark的平均分数）
    - 多个jsonl文件为每个benchmark详细的测评输出结果
    - 测试的配置文件（yaml）
4. 启动方式：
```
python ./eval/eval.py eval/eval_configs/xxx.yaml
```
5. 流程：
- step1：根据model_path和config的vllm配置参数，启动vllm服务
- 轮询vllm服务，当vllm服务可以访问后构建task_config依次测试
