pkill -9 -f "ray::"                                                                                                                       
pkill -9 -f "raylet"                                                                                                                      
pkill -9 -f "gcs_server"                                                                                                                  
rm -rf /tmp/ray_* 
pkill -f -u $(whoami) "verl|ray|cot_rewrite_grpo" && echo "已停止 verl 和 ray 相关进程"
pkill -f -u $(whoami) "python.*main_ppo\|vllm\|grpo" && echo "已停止训练相关进程"
rm -rf /tmp/ray_* && echo "已清理 Ray 临时目录"