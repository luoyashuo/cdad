from huggingface_hub import HfApi, upload_folder
import os

# 从环境变量获取 token，避免硬编码
token = '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'
# os.environ["HTTP_PROXY"]="http://127.0.0.1:17890"
# os.environ["HTTPS_PROXY"]="http://127.0.0.1:17890"

api = HfApi()

# 仓库配置
repo_id = "Yaseal/CDAD_EXP"
repo_type = "dataset"

# 上传三个文件夹到不同的路径，避免覆盖
upload_configs = [
    # {
    #     "folder_path": "/data/luoys/data/cdad/rewritten_output",
    #     "path_in_repo": "rewritten_output",
    # },
    # {
    #     "folder_path": "/data/luoys/data/cdad/omnithought_rv_optimal_top50k",
    #     "path_in_repo": "omnithought_rv_optimal_top50k",
    # },
    # {
    #     "folder_path": "/data/luoys/data/cdad/omnithought_rv_optimal_top20k",
    #     "path_in_repo": "omnithought_rv_optimal_top20k",
    # },
    {
        "folder_path": "/data/luoys/data/cdad/cold_start",
        "path_in_repo": "cold_start",
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
