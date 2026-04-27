import json
import os

def transform_data(config):
    # 获取文件路径
    input_path = config['rollout_file']
    output_dir = config['output_root']
    output_filename = config['output_filename']
    output_path = os.path.join(output_dir, output_filename)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # 改为逐行处理，节省内存且兼容 .jsonl
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                
                # 解析单行 JSON
                item = json.loads(line)
                
                question = item.get("question", "")
                rewrite_cots = item.get("rewrite_cots", [])
                
                # 提取第一个 rewrite_cot 的内容
                assistant_content = ""
                if rewrite_cots and len(rewrite_cots) > 0:
                    assistant_content = rewrite_cots[0].get("rewrite_cot", "")

                # 构建目标结构
                new_entry = {
                    "conversations": [
                        {
                            "role": "user",
                            "content": question
                        },
                        {
                            "role": "assistant",
                            "content": assistant_content
                        }
                    ]
                }
                
                # 将转换后的数据写入输出文件（每行一个 JSON）
                f_out.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
        
        print(f"转换完成！文件已按 JSONL 格式保存至: {output_path}")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")

# --- 配置示例 ---
config = {
    "rollout_file": "/data/hf_home/datasets/Yaseal/CDAD_EXP/Reformat_as_teacher/llama3_8b_3b_reformat_as_teacher/intermediate/rollout_output.jsonl",
    "output_root": "/data/hf_home/datasets/Yaseal/CDAD_EXP/Reformat_as_teacher",
    "output_filename": "rep_fsf_r80_30k_llama_reformat_as_teacher.jsonl"
}

if __name__ == "__main__":
    transform_data(config)