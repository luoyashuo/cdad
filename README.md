# cdad



# Installation

```
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install flash-attn --no-build-isolation
uv pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6
uv pip install git+https://github.com/NICTA/pyairports.git
uv pip install rich
uv pip install transformers==4.54.0 accelerate==1.8.0 deepspeed
uv pip install vllm==0.8.5.post1
sudo apt-get install python3.10-dev
uv pip install swanlab
uv pip install loguru
cd LlamaFactory
uv pip install llamafactory[metrics]
uv pip install -r requirements.txt
uv pip install evalscope
uv pip install evalscope[sandbox]
uv pip install vllm==0.8.5
uv pip install colorlog
uv pip install openpyxl

```

# Docker Installation

```bash
# 卸载旧版本 Docker
sudo apt-get purge -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras
sudo apt-get autoremove -y

# 安装 Docker CE
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 启动 Docker 服务
sudo service docker start

# 验证安装
docker --version
docker compose version

# 配置 Docker 数据目录到外部存储 (避免打爆主机存储)
sudo bash -c 'cat > /etc/docker/daemon.json << EOF
{
  "storage-driver": "vfs",
  "bridge": "none",
  "iptables": false,
  "data-root": "/mnt/moonfs/luoyashuo-m2/docker_data"
}
EOF'

# 重启 Docker 服务
sudo service docker restart
```

# Evaluation 临时文件配置

评测时的沙箱临时文件存储在 `/mnt/moonfs/luoyashuo-m2/tmp/evalscope_sandbox`，
通过 `eval_configs/eval_cg.yaml` 中的 `sandbox.work_dir` 配置。