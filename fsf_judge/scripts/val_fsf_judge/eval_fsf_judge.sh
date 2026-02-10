export CUDA_VISIBLE_DEVICES=3
python fsf_judge/fsf_judge_evaluate.py \
    --model_path /data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/fsf_judge/qwen3_8b_fsf_judge/checkpoint-12000 \
    --data_path /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/fsf_judge/val.jsonl \
    --batch_size 32 \
    --resume