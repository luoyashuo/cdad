#path = "/data/hf_home/datasets/Yaseal/CDAD_EXP/Reformat_as_teacher/rep_fsf_r80_30k_llama_reformat_as_teacher.jsonl"

#path = "/data/hf_home/datasets/Yaseal/CDAD_EXP/Reformat_as_teacher/rep_fsf_r80_30k_qwen_reformat_as_teacher.jsonl"

path = "/data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite"

with open(path, 'r', encoding='utf-8') as f:
    count = sum(1 for _ in f)

print("行数：", count)