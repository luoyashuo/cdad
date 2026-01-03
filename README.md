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
```