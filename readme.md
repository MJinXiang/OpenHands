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
