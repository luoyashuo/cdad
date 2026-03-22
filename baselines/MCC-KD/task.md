# 任务总览
使用deepmath_sft_30k数据集与Qwen2.5-7b-Instruct（LoRA）为基础，使用MCC-KD的方法来训练  
你需要使用的库  
transformers = 4.55.0  
accelerate==1.11.0  
deepspeed = 0.16.9  
peft==0.15.1  
flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl  
torch == 2.6.0+cu124  

## MCC-KD
论文位置：/home/luoys/qzq/mcc-deepmath/MCC-KD.pdf  
因为我已经获得了已有的数据集，所以你只需要实现
3.2  Rationale Filtering
3.3  Multi-CoT Consistent Distillation
和最终的
3.4  Overall Objective

## 数据集
原数据集位置/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train.jsonl  
你可以把新的数据集输出到/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train_mcckd.jsonl(文件需要新建)  
mjq -f /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train_mcckd.jsonl -s 1
格式对齐：
deepmath原格式：{"question": "...", "reasoning": [{"thought": "...","solution":"..."}, {"thought": "...","solution":"..."}, ...]}每个问题对应三个thought  

而且为了保持和主试验数据量一致，请在deepmath中随机选择15k个question，每个question按照MCC-KD的策略筛选两个thought作为最终的数据  
一个问题的reasoning应该遵循以下格式：<think>\n{thought}\n</think>\n\n{solution}


MCC-KD数据形式是 <r> Therefore, the answer is <a>，然后对着整个序列算Lce，对着答案<a>算Lkl（他的answer很小）
然后我们拼接的形式是<think>\n{thought}\n</think>\n\n{solution}  
solution是一大段话，其中有一个被\\box包裹的是答案
所以我们复现的时候把那个\\box里面的东西提取出来再算KL


## 训练
本机有8张GPU,每张GPU有40G的显存，你可以充分利用GPU算力，基于之前给出的库进行分布式训练
训练配置可以仿照/home/luoys/cdad/LlamaFactory/examples/vallina_sft/llama3_1b_vallina_sft_20k.yaml尽量和主试验保持对齐

## 备注
你不必试运行代码，因为本地还有其他实验，以免冲突。你只需确保代码逻辑正确无误，完美实现MCC-KD的核心流程即可  
另外，请最后在本目录下编写一份系统说明，告诉我你实现MCC-KD的核心方法的
请尽量利用现成的库进行编写
