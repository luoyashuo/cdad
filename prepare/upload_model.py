from huggingface_hub import HfApi, upload_folder
import os
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:17890"
os.environ['HTTP_PROXY'] = "http://127.0.0.1:17890"
os.environ["HUGGINGFACE_TOKEN"] = "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"
# 从环境变量获取 token，避免硬编码
token = os.environ.get('HUGGINGFACE_TOKEN')
if not token:
    raise ValueError("请先设置环境变量: export HUGGINGFACE_TOKEN=your_token_here")

api = HfApi()

# 仓库配置
repo_id = "Yaseal/CDAD_CKPT"
repo_type = "model"

# 上传三个文件夹到不同的路径，避免覆盖
upload_configs = [
    # {
    #     "folder_path": "/data/luoys/checkpoints/cdad/warmup/llama3_2_1b_rv_optimal_top20k",
    #     "path_in_repo": "llama3_2_1b_rv_optimal_top20k",
    # },
    # {
    #     "folder_path": "/data/luoys/checkpoints/cdad/warmup/llama3_2_1b_rv_optimal_top50k",
    #     "path_in_repo": "llama3_2_1b_rv_optimal_top50k",
    # },
    # {
    #     "folder_path": "/data/luoys/checkpoints/cdad/warmup/llama3_2_1b_rv_optimal_top50k_rewrite",
    #     "path_in_repo": "llama3_2_1b_rv_optimal_top50k_rewrite",
    # },
    # {
    #     "folder_path": "/data/luoys/checkpoints/cdad/warmup/llama3_2_1b_rv_optimal_top50k_rewrite_mix",
    #     "path_in_repo": "llama3_2_1b_rv_optimal_top50k_rewrite_mix",
    # },
    {
        "folder_path": "/data/luoys/checkpoints/cdad/cold_start/llama3_3b_instruct_top2k/checkpoint-182",
        "path_in_repo": "llama3_3b_instruct_top2k_cold_start",
    },
]


for config in upload_configs:
    print(f"正在上传 {config['folder_path']} -> {config['path_in_repo']}...")
    upload_folder(
        folder_path=config["folder_path"],
        path_in_repo=config["path_in_repo"],
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
    )
    print(f"上传完成: {config['path_in_repo']}")

print("所有文件夹上传完成！")
# nohup python /data/luoys/projects/cdad/prepare/upload_model.py > /data/luoys/projects/cdad/prepare/logs/upload.log 2>&1 &
