python eval.py eval_configs/eval_cg.yaml
python eval.py eval_configs/eval_rv.yaml
python eval.py eval_configs/eval_fsf.yaml
python eval.py eval_configs/eval_random.yaml


# nohup bash /home/luoyashuo/research/graph_kd/cdad/eval/launch_scripts/run.sh > /home/luoyashuo/research/graph_kd/cdad/eval/vllm_logs/run.log 2>&1 &
# nohup python eval.py eval_configs/eval_cg.yaml > /home/luoyashuo/research/graph_kd/cdad/eval/vllm_logs/run.log