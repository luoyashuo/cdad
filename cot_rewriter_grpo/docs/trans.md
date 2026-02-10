# 任务
目前的训练逻辑：scripts/grpo/start.sh
目前的实现很是单独在verl项目外面实现了src，但是我希望你把我自定义的reward计算逻辑直接集成到verl项目里面
# 要求
1. 你不可以修改数据配置，只能用目前提供好的数据格式
2. 所有模型都必须运行在GPU上
3. scripts/grpo/start.sh的运行前后的占卡逻辑不能够修改，但是你修改后仍然用scripts/grpo/start.sh来启动训练



/data-share/yeesuanAI08/lihaitao/luoys/data
rsync -avz --progress --partial --bwlimit=5120 \
  -e "ssh -o ServerAliveInterval=60 -o ForwardAgent=yes" \
  lihaitao@thuir5:/data-share/yeesuanAI08/lihaitao/luoys/data/cdad  \
  /Users/moonshot/Downloads/data 

/data-share/yeesuanAI08/lihaitao/luoys/data
rsync -avz --progress --partial --bwlimit=5120 \
  -e "ssh -o ServerAliveInterval=60 -o ForwardAgent=yes" \
  lihaitao@thuir5:/data-share/yeesuanAI08/lihaitao/luoys/projects/cdad \
  /Users/moonshot/Downloads/projects/cdad

  # 上传 data/cdad                                                                                                                        
rsync -avz --progress --partial --bwlimit=5120 \
    -e "ssh -o ServerAliveInterval=60 -o ForwardAgent=yes" \
    /Users/moonshot/Downloads/data/cdad/ \
    u20240075@RUC_4xA800_Direct:/home/u20240075/luoyashuo/data/cdad/                                                                      
                                                                                                                                          
  # 上传 projects/cdad                                                                                                                    
rsync -avz --progress --partial --bwlimit=5120 \                                                                           
-e "ssh -o ServerAliveInterval=60" \                                                                                                  
/Users/moonshot/Downloads/projects/cdad/cdad \                                                                                            
u20240075@RUC_4xA800_Direct:/home/u20240075/luoyashuo/projects/cdad/


scp -r -o ServerAliveInterval=60 -o ForwardAgent=yes \
    /Users/moonshot/Downloads/data \
    u20240075@RUC_4xA800_Direct:/home/u20240075/luoyashuo/data

scp -r -o ServerAliveInterval=60 -o ForwardAgent=yes \
    /Users/moonshot/Downloads/data/cdad/ \
    u20240075@RUC_4xA800_Direct:/home/u20240075/luoyashuo/projects

COPYFILE_DISABLE=1 tar czf - -C /Users/moonshot/Downloads data | \
    ssh -c aes128-gcm@openssh.com -o ServerAliveInterval=60 -o ForwardAgent=yes \
    u20240075@RUC_4xA800_Direct \
    "mkdir -p /home/u20240075/luoyashuo && cd /home/u20240075/luoyashuo && tar xzf -"

COPYFILE_DISABLE=1 tar czf - -C /Users/moonshot/Downloads/projects/cdad cdad | \
    ssh -c aes128-gcm@openssh.com -o ServerAliveInterval=60 -o ForwardAgent=yes \
    u20240075@RUC_4xA800_Direct \
    "cd /home/u20240075/luoyashuo && mkdir -p projects && tar xzf - && mv cdad/* projects/ 2>/dev/null; rm -rf cdad"