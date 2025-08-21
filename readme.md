# OpenHands 安装指南

##  安装步骤

### 1. 创建虚拟环境

#### 使用 conda
```bash
# 创建新的 conda 环境
conda create -n openhands python=3.12
conda activate openhands
```

### 2. 安装 OpenHands

```bash
# 安装 OpenHands 核心包
pip install openhands-ai
```

### 3. 安装 Mamba

Mamba 在没有权限时可以安装node.js：

```bash
# 通过 conda-forge 安装 mamba
conda install mamba -c conda-forge
```

### 4. 安装开发依赖

#### 安装 Node.js 和 Poetry
```bash
#使用 mamba
mamba install -c conda-forge nodejs=22
mamba install -c conda-forge poetry=1.8

# 验证安装
node --version
poetry --version
```

### 5. 构建项目

```bash
# 克隆项目
git clone https://github.com/All-Hands-AI/OpenHands.git
cd OpenHands

# 构建项目
make build
```

### 6. 安装 Docker

#### Linux (Ubuntu/Debian)
```bash
# Set required environment variables
export SANDBOX_VOLUMES="/netcache/mengjinxiang/Project/OpenHands:/workspace:rw"  # See

# Run OpenHands
docker run -it \
    --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.53-nikolaik \
    -e SANDBOX_USER_ID=$(id -u) \
    -e SANDBOX_VOLUMES=$SANDBOX_VOLUMES \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app-$(date +%Y%m%d%H%M%S) \
    docker.all-hands.dev/all-hands-ai/openhands:0.53 \

```


##  配置 OpenHands

### 创建配置文件
修改现存的config.template.toml的参数，同时改名为config.toml

不使用docker
```bash
[core]
# 工作区配置
workspace_base = "./workspace"
save_trajectory_path = "./trajectories"
file_store_path = "./results"
file_store = "local"
debug = true
enable_browser = false
runtime = "local"

[llm]
# 配置你的 LLM API
api_key = "your-api-key-here"
base_url = "https://api.openai.com/v1"
model = "gpt-4o"

[sandbox]
# 使用预构建的运行时镜像
volumes = "./workspace:/workspace:rw"

[agent]
enable_browsing = false
```


使用docker
```bash
[core]
# 工作区配置
workspace_base = "./workspace"
save_trajectory_path = "./trajectories"
file_store_path = "./results"
file_store = "local"
debug = true
enable_browser = false
runtime = "docker"

[llm]
# 配置你的 LLM API
api_key = "your-api-key-here"
base_url = "https://api.openai.com/v1"
model = "gpt-4o"

[sandbox]
# 使用预构建的运行时镜像
runtime_container_image = "docker.all-hands.dev/all-hands-ai/runtime:0.53-nikolaik"
volumes = "./workspace:/workspace:rw"

[agent]
enable_browsing = false
```


##  运行 OpenHands

```bash
# 运行简单任务
poetry run python -m openhands.core.main -t "write a hello world script in Python"
```


# 评估

## DA-Code
将数据文件放在evaluation/benchmarks/dacode/data下面
```bash
data/
├── configs/
├── gold/
└── source/
```

```bash
[llm]
# IMPORTANT: add your API key here, and set the model to the one you want to evaluate
model = "gpt-4o-2024-05-13"
api_key = "sk-XXX"

[llm.eval_gpt4_1106_preview_llm]
model = "gpt-4-1106-preview"
api_key = "XXX"
temperature = 0.0

[llm.eval_some_openai_compatible_model_llm]
model = "openai/MODEL_NAME"
base_url = "https://OPENAI_COMPATIBLE_URL/v1"
api_key = "XXX"
temperature = 0.0

#在config.toml中修改这个部分，在评估的时候只会用到这部分参数，其他的不会使用
[llm.gpt4o-eval]
model = "gpt-4o"
api_key = ""
base_url = ""
temperature = 0.0
top_p = 1.0
```

#### 运行第一个任务
```bash
bash evaluation/benchmarks/dacode/scripts/run_infer.sh llm.gpt4o-eval "" CodeActAgent 1
```

#### 运行前5个任务
```bash
bash evaluation/benchmarks/dacode/scripts/run_infer.sh llm.gpt4o-eval "" CodeActAgent 5
```

#### 跳过前3个任务，运行接下来的2个任务
```bash
bash evaluation/benchmarks/dacode/scripts/run_infer.sh llm.gpt4o-eval "" CodeActAgent 2 1 "" 3
```

#### 运行特定的任务
```bash
bash evaluation/benchmarks/dacode/scripts/run_infer.sh llm.gpt4o-eval "" CodeActAgent 0 1 "data-sa-001 plot-bar-015 di-csv-001"
```
