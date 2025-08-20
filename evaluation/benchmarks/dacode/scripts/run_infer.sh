# #!/usr/bin/env bash
# set -eo pipefail

# source "evaluation/utils/version_control.sh"

# MODEL_CONFIG=$1
# COMMIT_HASH=$2
# AGENT=$3
# EVAL_LIMIT=$4
# NUM_WORKERS=$5
# TASK_TYPE=$6  # 新增：特定任务类型过滤参数

# if [ -z "$NUM_WORKERS" ]; then
#   NUM_WORKERS=1
#   echo "Number of workers not specified, use default $NUM_WORKERS"
# fi

# checkout_eval_branch

# if [ -z "$AGENT" ]; then
#   echo "Agent not specified, use default CodeActAgent"
#   AGENT="CodeActAgent"
# fi

# get_openhands_version

# echo "AGENT: $AGENT"
# echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
# echo "MODEL_CONFIG: $MODEL_CONFIG"

# # 构建命令
# COMMAND="poetry run python evaluation/benchmarks/dacode/run_infer.py \
#   --agent-cls $AGENT \
#   --llm-config $MODEL_CONFIG \
#   --max-iterations 10 \
#   --eval-num-workers $NUM_WORKERS \
#   --eval-note $OPENHANDS_VERSION"

# # 评估限制参数
# if [ -n "$EVAL_LIMIT" ]; then
#   echo "EVAL_LIMIT: $EVAL_LIMIT"
#   COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
# fi

# # 增加任务类型过滤参数
# if [ -n "$TASK_TYPE" ]; then
#   echo "TASK_TYPE: $TASK_TYPE"
#   COMMAND="$COMMAND --task-type \"$TASK_TYPE\""
# fi

# # 记录时间戳
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# LOG_FILE="logs/dacode_eval_${TIMESTAMP}.log"

# echo "Starting dacode evaluation at $(date)"
# echo "Log will be saved to $LOG_FILE"

# # 运行命令并记录日志
# eval $COMMAND | tee $LOG_FILE

# echo "Evaluation completed at $(date)"





# #!/usr/bin/env bash
# # filepath: \netcache\mengjinxiang\Project\OpenHands\evaluation\benchmarks\dacode\scripts\run_infer.sh
# set -eo pipefail

# source "evaluation/utils/version_control.sh"

# MODEL_CONFIG=$1
# COMMIT_HASH=$2
# AGENT=$3
# EVAL_LIMIT=$4
# NUM_WORKERS=$5
# TASK_TYPE=$6

# if [ -z "$NUM_WORKERS" ]; then
#   NUM_WORKERS=1
#   echo "Number of workers not specified, use default $NUM_WORKERS"
# fi

# checkout_eval_branch

# if [ -z "$AGENT" ]; then
#   echo "Agent not specified, use default CodeActAgent"
#   AGENT="CodeActAgent"
# fi

# get_openhands_version

# echo "AGENT: $AGENT"
# echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
# echo "MODEL_CONFIG: $MODEL_CONFIG"

# # 创建日志目录
# mkdir -p logs

# # 构建命令 - 使用与 run_infer.py 兼容的参数
# COMMAND="poetry run python evaluation/benchmarks/dacode/run_infer.py \
#   -c $AGENT \
#   -l $MODEL_CONFIG \
#   -i 20"

# # 评估限制参数
# if [ -n "$EVAL_LIMIT" ]; then
#   echo "EVAL_LIMIT: $EVAL_LIMIT"
#   # 目前没有直接对应的参数，可以在需要时添加
# fi

# # 记录时间戳
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# LOG_FILE="logs/dacode_eval_${TIMESTAMP}.log"

# echo "Starting dacode evaluation at $(date)"
# echo "Log will be saved to $LOG_FILE"

# # 运行命令并记录日志
# eval $COMMAND | tee $LOG_FILE

# echo "Evaluation completed at $(date)"



#!/usr/bin/env bash
# filepath: \netcache\mengjinxiang\Project\OpenHands\evaluation\benchmarks\dacode\scripts\run_infer.sh
set -eo pipefail

source "evaluation/utils/version_control.sh"

MODEL_CONFIG=$1
COMMIT_HASH=$2
AGENT=$3
NUM_TASKS=$4      # 修改：使用具体任务数量而不是EVAL_LIMIT
NUM_WORKERS=$5    # 保留但目前未使用
TASK_IDS=$6       # 新增：特定任务ID参数
SKIP_NUM=$7       # 新增：跳过任务数量参数

# 设置默认值
if [ -z "$NUM_WORKERS" ]; then
  NUM_WORKERS=1
  echo "Number of workers not specified, use default $NUM_WORKERS"
fi

if [ -z "$AGENT" ]; then
  echo "Agent not specified, use default CodeActAgent"
  AGENT="CodeActAgent"
fi

if [ -z "$NUM_TASKS" ]; then
  echo "Number of tasks not specified, use default 1"
  NUM_TASKS=1
fi

if [ -z "$SKIP_NUM" ]; then
  SKIP_NUM=0
fi

checkout_eval_branch
get_openhands_version

echo "==================== DACODE EVALUATION SETUP ===================="
echo "AGENT: $AGENT"
echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "NUM_TASKS: $NUM_TASKS"
echo "SKIP_NUM: $SKIP_NUM"
if [ -n "$TASK_IDS" ]; then
  echo "SPECIFIC_TASK_IDS: $TASK_IDS"
fi
echo "=================================================================="

# 创建日志目录
mkdir -p logs

# 构建基础命令
COMMAND="poetry run python evaluation/benchmarks/dacode/run_infer.py \
  -c $AGENT \
  -l $MODEL_CONFIG \
  -i 20 \
  -n $NUM_TASKS"

# 添加跳过任务数量参数
if [ "$SKIP_NUM" -gt 0 ]; then
  echo "Skipping first $SKIP_NUM tasks"
  COMMAND="$COMMAND --skip-num $SKIP_NUM"
fi

# 添加特定任务ID参数
if [ -n "$TASK_IDS" ]; then
  echo "Processing specific task IDs: $TASK_IDS"
  COMMAND="$COMMAND --task-ids $TASK_IDS"
fi

# 记录时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/dacode_eval_${AGENT}_${NUM_TASKS}tasks_${TIMESTAMP}.log"

echo ""
echo "==================== EVALUATION COMPLETED ===================="
echo "Completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Log file: $LOG_FILE"

if [ $EXIT_CODE -eq 0 ]; then
  echo "Evaluation completed successfully!"

  echo ""
  echo "==================== STARTING RESULT EVALUATION ===================="

    EVAL_COMMAND="python evaluation/benchmarks/dacode/evaluate.py "
    echo "Evaluation command: $EVAL_COMMAND"

    eval $EVAL_COMMAND
    EVAL_EXIT_CODE=$?

    if [ $EVAL_EXIT_CODE -eq 0 ]; then
      echo "Result evaluation completed successfully!"
    else
      echo "Result evaluation failed with exit code $EVAL_EXIT_CODE"
    fi
  else
    echo "Warning: Could not find output directory for evaluation"
    echo "You can manually run: python evaluation/benchmarks/dacode/evaluate.py"
  fi

  echo "=================================================================="
else
  echo "Evaluation failed with exit code $EXIT_CODE"
  echo "Check the log file for details: $LOG_FILE"
fi

echo "=================================================================="

exit $EXIT_CODE
