# import asyncio
# import json
# import os
# import pathlib
# import re
# import subprocess
# import shutil
# import zipfile
# from typing import Any, Dict, List, Optional, Union

# import pandas as pd
# import numpy as np
# from PIL import Image
# # from func_timeout import FunctionTimedOut, func_timeout
# from tqdm import tqdm

# from openhands.llm.llm_registry import LLMRegistry

# from evaluation.utils.shared import (
#     EvalMetadata,
#     EvalOutput,
#     compatibility_for_eval_history_pairs,
#     get_default_sandbox_config_for_eval,
#     make_metadata,
#     prepare_dataset,
#     reset_logger_for_multiprocessing,
#     run_evaluation,
# )
# from openhands.controller.state.state import State
# from openhands.core.config import (
#     OpenHandsConfig,
#     get_llm_config_arg,
#     parse_arguments,
# )
# from openhands.core.logger import openhands_logger as logger
# from openhands.core.main import create_runtime, run_controller
# from openhands.events.action import CmdRunAction, MessageAction
# from openhands.events.observation import CmdOutputObservation
# from openhands.runtime.base import Runtime
# from openhands.utils.async_utils import call_async_from_sync


# def codeact_user_response(state: State) -> str:
#     """Define how to respond to the agent during task execution."""
#     msg = (
#         'Please continue working on the task using the approach you think is suitable.\n'
#         'If you think you have completed the task, please finish the interaction using the "finish" tool.\n'
#         'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP OR USE THE INTERNET TO SOLVE THIS TASK.\n'
#     )
#     if state.history:
#         # Check if the agent has tried to talk to the user 3 times, if so, let the agent know it can give up
#         user_msgs = [
#             event
#             for event in state.history
#             if isinstance(event, MessageAction) and event.source == 'user'
#         ]
#         if len(user_msgs) > 2:
#             # Let the agent know that it can give up when it has tried 3 times
#             return (
#                 msg
#                 + 'If you want to give up, use the "finish" tool to finish the interaction.\n'
#             )
#     return msg


# AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
#     'CodeActAgent': codeact_user_response,
# }

# AGENT_CLS_TO_INST_SUFFIX = {
#     'CodeActAgent': 'When you think you have completed the task, please finish the interaction using the "finish" tool.\n'
# }


# def get_config(
#     metadata: EvalMetadata,
# ) -> OpenHandsConfig:
#     """Configure the OpenHands environment for dacode tasks."""
#     sandbox_config = get_default_sandbox_config_for_eval()
#     # Using data science oriented container with Python libraries
#     sandbox_config.runtime_container_image = "docker.all-hands.dev/all-hands-ai/runtime:0.53-nikolaik"
#     # sandbox_config.base_container_image = 'python:3.11-bullseye'

#     config = OpenHandsConfig(
#         default_agent=metadata.agent_class,
#         run_as_openhands=False,
#         runtime='docker',
#         max_iterations=metadata.max_iterations,
#         sandbox=sandbox_config,
#         # Do not mount workspace
#         workspace_base=None,
#         workspace_mount_path=None,
#     )
#     config.set_llm_config(metadata.llm_config)
#     agent_config = config.get_agent_config(metadata.agent_class)
#     agent_config.enable_prompt_extensions = False
#     return config


# LOCAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data')
# SOURCE_DATA_PATH = os.path.join(LOCAL_DATASET_PATH, 'source')
# CONFIG_PATH = os.path.join(LOCAL_DATASET_PATH, 'configs')


# def load_dacode_dataset():
#     """Load and prepare the dacode dataset."""
#     # Load task configs
#     task_config_path = os.path.join(CONFIG_PATH, 'task', 'examples.jsonl')
#     if not os.path.exists(task_config_path):
#         logger.error(f"Task config file not found: {task_config_path}")
#         raise FileNotFoundError(f"Task config file not found: {task_config_path}")

#     # Read task configurations
#     tasks = []
#     with open(task_config_path, 'r') as f:
#         for line in f:
#             if line.strip():
#                 task = json.loads(line)
#                 tasks.append(task)

#     # Create dataset
#     dataset = pd.DataFrame(tasks)

#     # Add source data paths
#     dataset['source_dir'] = dataset['id'].apply(
#         lambda x: os.path.join(SOURCE_DATA_PATH, x)
#     )

#     dataset['instance_id'] = dataset['id']

#     return dataset


# def get_task_files(task_dir: str) -> Dict[str, str]:
#     """Get all files in a task directory with their contents."""
#     files = {}
#     if not os.path.exists(task_dir):
#         logger.warning(f"Task directory does not exist: {task_dir}")
#         return files

#     for filename in os.listdir(task_dir):
#         filepath = os.path.join(task_dir, filename)
#         if os.path.isfile(filepath):
#             try:
#                 with open(filepath, 'r', encoding='utf-8') as f:
#                     content = f.read()
#                 files[filename] = content
#             except UnicodeDecodeError:
#                 # Binary file, just note its existence
#                 files[filename] = "[BINARY FILE]"

#     return files


# def create_task_prompt(instance: pd.Series) -> str:
#     """Create a prompt for the dacode task."""
#     task_id = instance['id']
#     task_type = instance['type']
#     instruction = instance['instruction']
#     hardness = instance['hardness']

#     # Get additional files
#     task_files = get_task_files(instance['source_dir'])

#     # Build the prompt
#     prompt = f"# Task: {task_id}\n\n"
#     prompt += f"## Type: {task_type}\n\n"
#     prompt += f"## Difficulty: {hardness}\n\n"
#     prompt += f"## Instruction:\n{instruction}\n\n"

#     # Add information about available files
#     if task_files:
#         prompt += "## Available Files:\n\n"
#         for filename, content in task_files.items():
#             if content == "[BINARY FILE]":
#                 prompt += f"- {filename} (binary file)\n"
#             elif filename.endswith(('.csv', '.json', '.txt', '.md')):
#                 prompt += f"- {filename}\n"

#         # Add content of important instruction files
#         for filename, content in task_files.items():
#             if filename.endswith(('.md', '.txt')) and "sample" not in filename.lower() and content != "[BINARY FILE]":
#                 prompt += f"\n### Content of {filename}:\n```\n{content}\n```\n"

#     return prompt


# def initialize_runtime(
#     runtime: Runtime,
#     instance: pd.Series,
# ):
#     """Initialize the runtime environment for the dacode task."""
#     logger.info(f'{"-" * 50} BEGIN Runtime Initialization Fn {"-" * 50}')
#     obs: CmdOutputObservation

#     # Copy task files to workspace
#     task_dir = instance['source_dir']
#     if os.path.exists(task_dir):
#         for filename in os.listdir(task_dir):
#             filepath = os.path.join(task_dir, filename)
#             if os.path.isfile(filepath):
#                 runtime.copy_to(filepath, f'/workspace/{filename}')

#     # Install necessary packages based on task type
#     task_type = instance['type']

#     # Common packages for all tasks
#     packages = ['pandas', 'numpy', 'scipy']

#     # Add task-specific packages
#     if 'ML' in task_type:
#         packages.extend(['scikit-learn', 'joblib'])
#     if 'Visualization' in task_type:
#         packages.extend(['matplotlib', 'seaborn'])
#     if 'Statistical Analysis' in task_type:
#         packages.extend(['statsmodels'])

#     # Install packages
#     install_cmd = f"pip install --no-cache-dir {' '.join(packages)}"
#     action = CmdRunAction(command=install_cmd)
#     obs = runtime.run_action(action)
#     logger.info(obs, extra={'msg_type': 'OBSERVATION'})

#     # Check if workspace is properly set up
#     action = CmdRunAction(command='cd /workspace && ls -la')
#     obs = runtime.run_action(action)
#     logger.info(obs, extra={'msg_type': 'OBSERVATION'})

#     logger.info(f'{"-" * 50} END Runtime Initialization Fn {"-" * 50}')


# def evaluate_result(
#     runtime: Runtime,
#     instance: pd.Series,
#     timeout: int = 30
# ) -> Dict[str, Any]:
#     """Evaluate the agent's solution for dacode tasks."""
#     task_id = instance['id']
#     task_type = instance['type']
#     post_process = instance.get('post_process', [])

#     result = {
#         'result': {'passed': 0, 'status': 'unknown'},
#         'metadata': {'task_type': task_type}
#     }

#     try:
#         # Check if expected output files exist based on task type
#         if 'ML' in task_type or 'Statistical Analysis' in task_type or 'Data Manipulation' in task_type:
#             # These tasks typically produce CSV files
#             expected_outputs = []

#             # Extract expected output filename from instruction
#             instruction = instance['instruction']
#             filename_matches = re.findall(r'(?:save|write).*?(?:to|into|in).*?[\'"]([^\'"]*.csv)[\'"]', instruction, re.IGNORECASE)
#             if filename_matches:
#                 expected_outputs.extend(filename_matches)
#             else:
#                 # Default output files to check
#                 expected_outputs = ['result.csv', 'submission.csv', 'output.csv']

#             # Check if any expected output file exists
#             for output_file in expected_outputs:
#                 action = CmdRunAction(command=f'ls -la /workspace/{output_file} 2>/dev/null || echo "File not found"')
#                 obs = runtime.run_action(action)
#                 if "File not found" not in obs.content:
#                     # File exists, check content
#                     action = CmdRunAction(command=f'cat /workspace/{output_file} 2>/dev/null || echo "Cannot read file"')
#                     obs = runtime.run_action(action)
#                     if "Cannot read file" not in obs.content and len(obs.content.strip()) > 0:
#                         result['result']['passed'] = 1
#                         result['result']['status'] = 'success'
#                         result['metadata']['output_file'] = output_file
#                         result['metadata']['output_content'] = obs.content
#                         break

#         elif 'Data Visualization' in task_type:
#             # For visualization tasks, check for image files
#             expected_formats = ['png', 'jpg', 'jpeg']
#             for fmt in expected_formats:
#                 action = CmdRunAction(command=f'ls -la /workspace/*.{fmt} 2>/dev/null || echo "File not found"')
#                 obs = runtime.run_action(action)
#                 if "File not found" not in obs.content:
#                     # Image file exists
#                     for line in obs.content.splitlines():
#                         if f'.{fmt}' in line:
#                             img_file = line.split()[-1]
#                             result['result']['passed'] = 1
#                             result['result']['status'] = 'success'
#                             result['metadata']['output_file'] = img_file
#                             break
#                     if result['result']['passed'] == 1:
#                         break

#         elif 'Data Insight' in task_type:
#             # For data insights, look for JSON or text output
#             action = CmdRunAction(command='find /workspace -type f -name "*.json" -o -name "output.txt" 2>/dev/null || echo "File not found"')
#             obs = runtime.run_action(action)
#             if "File not found" not in obs.content and obs.content.strip():
#                 output_file = obs.content.strip().splitlines()[0]
#                 action = CmdRunAction(command=f'cat {output_file} 2>/dev/null || echo "Cannot read file"')
#                 obs = runtime.run_action(action)
#                 if "Cannot read file" not in obs.content and len(obs.content.strip()) > 0:
#                     result['result']['passed'] = 1
#                     result['result']['status'] = 'success'
#                     result['metadata']['output_file'] = output_file
#                     result['metadata']['output_content'] = obs.content

#         # If no output files found yet, check for any Python script that might have been created
#         if result['result']['passed'] == 0:
#             action = CmdRunAction(command='find /workspace -name "*.py" -type f | grep -v "__pycache__" || echo "No Python files"')
#             obs = runtime.run_action(action)
#             if "No Python files" not in obs.content and obs.content.strip():
#                 # Python file exists, let's look at its content
#                 python_files = obs.content.strip().split('\n')
#                 for py_file in python_files:
#                     action = CmdRunAction(command=f'cat {py_file} 2>/dev/null || echo "Cannot read file"')
#                     obs = runtime.run_action(action)
#                     if "Cannot read file" not in obs.content and len(obs.content.strip()) > 0:
#                         result['metadata']['python_file'] = py_file
#                         result['metadata']['python_content'] = obs.content
#                         # The existence of a Python file is a partial success
#                         if result['result']['status'] == 'unknown':
#                             result['result']['status'] = 'partial'
#                             result['result']['passed'] = 0.5

#                 # If we have a Python file, try running it to see if it produces output
#                 if 'python_file' in result['metadata']:
#                     action = CmdRunAction(command=f'cd /workspace && python {os.path.basename(result["metadata"]["python_file"])} 2>&1 || echo "Execution failed"')
#                     obs = runtime.run_action(action)
#                     result['metadata']['execution_output'] = obs.content
#                     if "Execution failed" not in obs.content:
#                         # Check again for output files after execution
#                         action = CmdRunAction(command='ls -la /workspace/')
#                         obs = runtime.run_action(action)
#                         result['metadata']['workspace_after_execution'] = obs.content
#                         # Re-evaluate success criteria
#                         return evaluate_result(runtime, instance, timeout)

#     # except FunctionTimedOut:
#     #     result['result']['status'] = 'timeout'
#     except Exception as e:
#         if "timeout" in str(e).lower():
#             result['result']['status'] = 'timeout'
#         else:
#             result['result']['status'] = 'error'
#         result['metadata']['error'] = str(e)
#         # logger.error(f'Error in evaluate_result: {e}')
#         # result['result']['status'] = 'error'
#         # result['metadata']['error'] = str(e)

#     return result


# def complete_runtime(
#     runtime: Runtime,
#     instance: pd.Series,
# ) -> dict[str, Any]:
#     """Complete the runtime for the dacode task and evaluate results."""
#     logger.info(f'{"-" * 50} BEGIN Runtime Completion Fn {"-" * 50}')
#     timeout = 60  # Longer timeout for data processing tasks

#     # Evaluate the result
#     test_result = evaluate_result(runtime, instance, timeout)

#     logger.info(f'{"-" * 50} END Runtime Completion Fn {"-" * 50}')
#     return test_result


# def process_instance(
#     instance: pd.Series,
#     metadata: EvalMetadata,
#     reset_logger: bool = True,
# ) -> EvalOutput:
#     """Process a single dacode task instance."""
#     config = get_config(metadata)
#     instance_id = instance['id']

#     # Set up the logger
#     if reset_logger:
#         log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
#         reset_logger_for_multiprocessing(logger, instance_id, log_dir)
#     else:
#         logger.info(f'Starting evaluation for instance {instance_id}.')

#     # Create instruction prompt
#     instruction = create_task_prompt(instance)

#     instruction += (
#         '\n\nIMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
#         'You should analyze the provided files, write the necessary code, and save results in the required format.\n'
#     )

#     # Add agent-specific suffix
#     instruction += AGENT_CLS_TO_INST_SUFFIX[metadata.agent_class]

#     # Create and initialize runtime
#     # runtime = create_runtime(config)
#     llm_registry = LLMRegistry(config)
#     runtime = create_runtime(config, llm_registry)
#     call_async_from_sync(runtime.connect)
#     initialize_runtime(runtime, instance)

#     # Run the agent
#     state: State | None = asyncio.run(
#         run_controller(
#             config=config,
#             initial_user_action=MessageAction(content=instruction),
#             fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[metadata.agent_class],
#             runtime=runtime,
#         )
#     )

#     # Evaluate the agent's solution
#     test_result = complete_runtime(runtime, instance)

#     # Prepare output
#     if state is None:
#         raise ValueError('State should not be None.')

#     metrics = state.metrics.get() if state.metrics else None
#     histories = compatibility_for_eval_history_pairs(state.history)

#     output = EvalOutput(
#         instance_id=instance_id,
#         instruction=instruction,
#         metadata=metadata,
#         history=histories,
#         metrics=metrics,
#         error=state.last_error if state and state.last_error else None,
#         test_result=test_result,
#     )

#     return output


# if __name__ == '__main__':
#     args = parse_arguments()

#     # Load dacode dataset
#     dataset = load_dacode_dataset()

#     # Set up LLM config
#     llm_config = None
#     if args.llm_config:
#         llm_config = get_llm_config_arg(args.llm_config)
#         # modify_params must be False for evaluation purpose
#         llm_config.modify_params = False
#     if llm_config is None:
#         raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

#     # Create metadata
#     metadata = make_metadata(
#         llm_config,
#         'DACODE',
#         args.agent_cls,
#         args.max_iterations,
#         os.environ.get('OPENHANDS_VERSION', 'v0.53.0'),
#         os.path.join('evaluation_output', 'dacode')
#         # args.eval_note,
#         # args.eval_output_dir,
#     )

#     # Set up output
#     output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
#     instances = prepare_dataset(dataset, output_file, None)  # Do not limit the number of instances for evaluation

#     # Run evaluation
#     run_evaluation(
#         instances, metadata, output_file, 1, process_instance
#     )







# import asyncio
# import json
# import os
# import pathlib
# import re
# import subprocess
# import shutil
# import zipfile
# from typing import Any, Dict, List, Optional, Union

# import pandas as pd
# import numpy as np
# from PIL import Image
# from tqdm import tqdm

# from openhands.llm.llm_registry import LLMRegistry

# from evaluation.utils.shared import (
#     EvalMetadata,
#     EvalOutput,
#     compatibility_for_eval_history_pairs,
#     get_default_sandbox_config_for_eval,
#     make_metadata,
#     prepare_dataset,
#     reset_logger_for_multiprocessing,
#     run_evaluation,
# )
# from openhands.controller.state.state import State
# from openhands.core.config import (
#     OpenHandsConfig,
#     get_llm_config_arg,
#     parse_arguments,
# )
# from openhands.core.logger import openhands_logger as logger
# from openhands.core.main import create_runtime, run_controller
# from openhands.events.action import CmdRunAction, MessageAction
# from openhands.events.observation import CmdOutputObservation
# from openhands.runtime.base import Runtime
# from openhands.utils.async_utils import call_async_from_sync


# def codeact_user_response(state: State) -> str:
#     """Define how to respond to the agent during task execution."""
#     msg = (
#         'Please continue working on the task using the approach you think is suitable.\n'
#         'If you think you have completed the task, please finish the interaction using the "finish" tool.\n'
#         'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP OR USE THE INTERNET TO SOLVE THIS TASK.\n'
#     )
#     if state.history:
#         # Check if the agent has tried to talk to the user 3 times, if so, let the agent know it can give up
#         user_msgs = [
#             event
#             for event in state.history
#             if isinstance(event, MessageAction) and event.source == 'user'
#         ]
#         if len(user_msgs) > 2:
#             # Let the agent know that it can give up when it has tried 3 times
#             return (
#                 msg
#                 + 'If you want to give up, use the "finish" tool to finish the interaction.\n'
#             )
#     return msg


# AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
#     'CodeActAgent': codeact_user_response,
# }

# AGENT_CLS_TO_INST_SUFFIX = {
#     'CodeActAgent': 'When you think you have completed the task, please finish the interaction using the "finish" tool.\n'
# }


# def get_config(
#     metadata: EvalMetadata,
# ) -> OpenHandsConfig:
#     """Configure the OpenHands environment for dacode tasks."""
#     sandbox_config = get_default_sandbox_config_for_eval()
#     # Using data science oriented container with Python libraries
#     sandbox_config.runtime_container_image = "docker.all-hands.dev/all-hands-ai/runtime:0.53-nikolaik"

#     config = OpenHandsConfig(
#         default_agent=metadata.agent_class,
#         run_as_openhands=False,
#         runtime='docker',
#         max_iterations=metadata.max_iterations,
#         sandbox=sandbox_config,
#         # Do not mount workspace
#         workspace_base=None,
#         workspace_mount_path=None,
#     )
#     config.set_llm_config(metadata.llm_config)
#     agent_config = config.get_agent_config(metadata.agent_class)
#     agent_config.enable_prompt_extensions = False
#     return config


# LOCAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data')
# SOURCE_DATA_PATH = os.path.join(LOCAL_DATASET_PATH, 'source')
# CONFIG_PATH = os.path.join(LOCAL_DATASET_PATH, 'configs')


# def load_dacode_dataset():
#     """Load and prepare the dacode dataset."""
#     # Load task configs
#     task_config_path = os.path.join(CONFIG_PATH, 'task', 'examples.jsonl')
#     if not os.path.exists(task_config_path):
#         logger.error(f"Task config file not found: {task_config_path}")
#         raise FileNotFoundError(f"Task config file not found: {task_config_path}")

#     # Read task configurations
#     tasks = []
#     with open(task_config_path, 'r') as f:
#         for line in f:
#             if line.strip():
#                 task = json.loads(line)
#                 tasks.append(task)

#     logger.info(f"Loaded {len(tasks)} tasks from {task_config_path}")

#     # Create dataset
#     dataset = pd.DataFrame(tasks)

#     # Add source data paths
#     dataset['source_dir'] = dataset['id'].apply(
#         lambda x: os.path.join(SOURCE_DATA_PATH, x)
#     )

#     dataset['instance_id'] = dataset['id']

#     return dataset


# def get_task_files(task_dir: str) -> Dict[str, str]:
#     """Get all files in a task directory with their contents."""
#     files = {}
#     if not os.path.exists(task_dir):
#         logger.warning(f"Task directory does not exist: {task_dir}")
#         return files

#     for filename in os.listdir(task_dir):
#         filepath = os.path.join(task_dir, filename)
#         if os.path.isfile(filepath):
#             try:
#                 with open(filepath, 'r', encoding='utf-8') as f:
#                     content = f.read()
#                 files[filename] = content
#             except UnicodeDecodeError:
#                 # Binary file, just note its existence
#                 files[filename] = "[BINARY FILE]"

#     return files


# def create_task_prompt(instance: pd.Series) -> str:
#     """Create a prompt for the dacode task."""
#     task_id = instance['id']
#     task_type = instance['type']
#     instruction = instance['instruction']
#     hardness = instance['hardness']

#     # Get additional files
#     task_files = get_task_files(instance['source_dir'])

#     # Build the prompt
#     prompt = f"# Task: {task_id}\n\n"
#     prompt += f"## Type: {task_type}\n\n"
#     prompt += f"## Difficulty: {hardness}\n\n"
#     prompt += f"## Instruction:\n{instruction}\n\n"

#     # Add information about available files
#     if task_files:
#         prompt += "## Available Files:\n\n"
#         for filename, content in task_files.items():
#             if content == "[BINARY FILE]":
#                 prompt += f"- {filename} (binary file)\n"
#             elif filename.endswith(('.csv', '.json', '.txt', '.md')):
#                 prompt += f"- {filename}\n"

#         # Add content of important instruction files
#         for filename, content in task_files.items():
#             if filename.endswith(('.md', '.txt')) and "sample" not in filename.lower() and content != "[BINARY FILE]":
#                 prompt += f"\n### Content of {filename}:\n```\n{content}\n```\n"

#     return prompt


# def initialize_runtime(
#     runtime: Runtime,
#     instance: pd.Series,
# ):
#     """Initialize the runtime environment for the dacode task."""
#     logger.info(f'{"-" * 50} BEGIN Runtime Initialization Fn {"-" * 50}')
#     obs: CmdOutputObservation

#     # Copy task files to workspace
#     task_dir = instance['source_dir']
#     if os.path.exists(task_dir):
#         for filename in os.listdir(task_dir):
#             filepath = os.path.join(task_dir, filename)
#             if os.path.isfile(filepath):
#                 runtime.copy_to(filepath, f'/workspace/{filename}')

#     # Install necessary packages based on task type
#     task_type = instance['type']

#     # Common packages for all tasks
#     packages = ['pandas', 'numpy', 'scipy']

#     # Add task-specific packages
#     if 'ML' in task_type:
#         packages.extend(['scikit-learn', 'joblib'])
#     if 'Visualization' in task_type:
#         packages.extend(['matplotlib', 'seaborn'])
#     if 'Statistical Analysis' in task_type:
#         packages.extend(['statsmodels'])

#     # Install packages
#     install_cmd = f"pip install --no-cache-dir {' '.join(packages)}"
#     action = CmdRunAction(command=install_cmd)
#     obs = runtime.run_action(action)
#     logger.info(obs, extra={'msg_type': 'OBSERVATION'})

#     # Check if workspace is properly set up
#     action = CmdRunAction(command='cd /workspace && ls -la')
#     obs = runtime.run_action(action)
#     logger.info(obs, extra={'msg_type': 'OBSERVATION'})

#     logger.info(f'{"-" * 50} END Runtime Initialization Fn {"-" * 50}')


# def save_task_results(instance_id: str, output: EvalOutput, metadata: EvalMetadata, runtime: Runtime):
#     """Save task results to individual folders."""
#     task_output_dir = os.path.join(metadata.eval_output_dir, f'task_{instance_id}')
#     os.makedirs(task_output_dir, exist_ok=True)

#     # Save task result as JSON
#     result_file = os.path.join(task_output_dir, 'result.json')
#     result_data = {
#         'instance_id': output.instance_id,
#         'instruction': output.instruction,
#         'test_result': output.test_result,
#         'metrics': output.metrics,
#         'error': output.error,
#         'history_length': len(output.history) if output.history else 0
#     }

#     with open(result_file, 'w', encoding='utf-8') as f:
#         json.dump(result_data, f, indent=2, ensure_ascii=False)

#     # Copy workspace files to task folder
#     workspace_files_dir = os.path.join(task_output_dir, 'workspace_files')
#     os.makedirs(workspace_files_dir, exist_ok=True)

#     try:
#         # List all files in workspace
#         action = CmdRunAction(command='find /workspace -type f -name "*" | head -50')
#         obs = runtime.run_action(action)

#         if obs.content.strip():
#             workspace_files = obs.content.strip().split('\n')

#             for file_path in workspace_files:
#                 if file_path.startswith('/workspace/'):
#                     filename = os.path.basename(file_path)
#                     if filename and not filename.startswith('.'):  # Skip hidden files
#                         try:
#                             local_path = os.path.join(workspace_files_dir, filename)
#                             read_action = CmdRunAction(command=f'cat "{file_path}"')
#                             read_obs = runtime.run_action(read_action)

#                             if read_obs.exit_code == 0:
#                                 with open(local_path, 'w', encoding='utf-8', errors='ignore') as f:
#                                     f.write(read_obs.content)
#                                 logger.info(f"Copied workspace file: {filename}")
#                             else:
#                                 b64_action = CmdRunAction(command=f'base64 "{file_path}"')
#                                 b64_obs = runtime.run_action(b64_action)

#                                 if b64_obs.exit_code == 0:
#                                     import base64
#                                     try:
#                                         file_content = base64.b64decode(b64_obs.content.strip())
#                                         with open(local_path, 'wb') as f:
#                                             f.write(file_content)
#                                         logger.info(f"Copied binary file: {filename}")
#                                     except Exception as e:
#                                         logger.warning(f"Failed to decode binary file {filename}: {e}")

#                         except Exception as e:
#                             logger.warning(f"Failed to copy file {filename}: {e}")

#     except Exception as e:
#         logger.warning(f"Failed to copy workspace files for task {instance_id}: {e}")

#     try:
#         action = CmdRunAction(command='ls -la /workspace/')
#         obs = runtime.run_action(action)
#         workspace_list_file = os.path.join(task_output_dir, 'workspace_listing.txt')
#         with open(workspace_list_file, 'w', encoding='utf-8') as f:
#             f.write(obs.content)
#         logger.info(f"Saved workspace listing to: workspace_listing.txt")
#     except Exception as e:
#         logger.warning(f"Failed to save workspace listing: {e}")

#     logger.info(f"Task results saved to: {task_output_dir}")
#     return task_output_dir


# def process_instance(
#     instance: pd.Series,
#     metadata: EvalMetadata,
#     reset_logger: bool = True,
# ) -> EvalOutput:
#     """Process a single dacode task instance - Execute only, no evaluation."""
#     config = get_config(metadata)
#     instance_id = instance['id']

#     # Set up the logger
#     if reset_logger:
#         log_dir = os.path.join(metadata.eval_output_dir, 'logs')
#         reset_logger_for_multiprocessing(logger, instance_id, log_dir)
#     else:
#         logger.info(f'Starting execution for instance {instance_id}.')

#     # Create instruction prompt
#     instruction = create_task_prompt(instance)

#     instruction += (
#         '\n\nIMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
#         'You should analyze the provided files, write the necessary code, and save results in the required format.\n'
#     )

#     # Add agent-specific suffix
#     instruction += AGENT_CLS_TO_INST_SUFFIX[metadata.agent_class]

#     # Create and initialize runtime
#     llm_registry = LLMRegistry(config)
#     runtime = create_runtime(config, llm_registry)
#     call_async_from_sync(runtime.connect)
#     initialize_runtime(runtime, instance)

#     # Run the agent
#     state: State | None = asyncio.run(
#         run_controller(
#             config=config,
#             initial_user_action=MessageAction(content=instruction),
#             fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[metadata.agent_class],
#             runtime=runtime,
#         )
#     )

#     # Prepare output (no evaluation)
#     if state is None:
#         raise ValueError('State should not be None.')

#     metrics = state.metrics.get() if state.metrics else None
#     histories = compatibility_for_eval_history_pairs(state.history)

#     # Create a simple completion result instead of evaluation
#     completion_result = {
#         'result': {'status': 'completed'},
#         'metadata': {'task_type': instance['type']}
#     }

#     output = EvalOutput(
#         instance_id=instance_id,
#         instruction=instruction,
#         metadata=metadata,
#         history=histories,
#         metrics=metrics,
#         error=state.last_error if state and state.last_error else None,
#         test_result=completion_result,
#     )

#     # Save results to individual task folder
#     save_task_results(instance_id, output, metadata, runtime)

#     logger.info(f'Task execution completed for instance {instance_id}')
#     return output


# def prepare_dataset_custom(
#     dataset: pd.DataFrame,
#     output_file: str,
#     eval_n_limit: int,
#     eval_ids: list[str] | None = None,
#     skip_num: int | None = None,
# ):
#     assert 'instance_id' in dataset.columns, (
#         "Expected 'instance_id' column in the dataset."
#     )

#     logger.info(f'Preparing dataset with {len(dataset)} total instances')

#     if eval_ids:
#         eval_ids_converted = [dataset['instance_id'].dtype.type(id) for id in eval_ids]
#         dataset = dataset[dataset['instance_id'].isin(eval_ids_converted)]
#         logger.info(f'Limiting execution to {len(eval_ids)} specific instances.')
#     elif skip_num and skip_num >= 0:
#         skip_num = min(skip_num, len(dataset))
#         dataset = dataset.iloc[skip_num:]
#         logger.info(f'Starting execution with skipping first {skip_num} instances ({len(dataset)} instances to run).')
#         if eval_n_limit and eval_n_limit > 0:
#             dataset = dataset.head(eval_n_limit)
#             # dataset = dataset.sample(
#             #     min(eval_n_limit, len(dataset)), random_state=42, replace=False
#             # )
#             logger.info(f'Randomly sampling {eval_n_limit} unique instances with random seed 42.')
#     elif eval_n_limit and eval_n_limit > 0:
#         dataset = dataset.head(eval_n_limit)
#         # dataset = dataset.sample(
#         #     min(eval_n_limit, len(dataset)), random_state=42, replace=False
#         # )
#         logger.info(f'Randomly sampling {eval_n_limit} unique instances with random seed 42.')

#     def make_serializable(instance_dict: dict) -> dict:
#         import numpy as np
#         for k, v in instance_dict.items():
#             if isinstance(v, np.ndarray):
#                 instance_dict[k] = v.tolist()
#             elif isinstance(v, pd.Timestamp):
#                 instance_dict[k] = str(v)
#             elif isinstance(v, dict):
#                 instance_dict[k] = make_serializable(v)
#         return instance_dict

#     new_dataset = [
#         make_serializable(instance.to_dict())
#         for _, instance in dataset.iterrows()
#     ]

#     logger.info(f'Total instances to process: {len(new_dataset)}')
#     return pd.DataFrame(new_dataset)


# if __name__ == '__main__':
#     args = parse_arguments()

#     # Load dacode dataset
#     dataset = load_dacode_dataset()

#     if dataset.empty:
#         logger.error("No tasks found in dataset!")
#         exit(1)

#     # Set up LLM config
#     llm_config = None
#     if args.llm_config:
#         llm_config = get_llm_config_arg(args.llm_config)
#         # modify_params must be False for evaluation purpose
#         llm_config.modify_params = False
#     if llm_config is None:
#         raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

#     # Create metadata
#     metadata = make_metadata(
#         llm_config,
#         'DACODE',
#         args.agent_cls,
#         args.max_iterations,
#         os.environ.get('OPENHANDS_VERSION', 'v0.53.0'),
#         os.path.join('evaluation_output', 'dacode')
#     )

#     # Set up output
#     output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')

#     instances = prepare_dataset_custom(dataset, output_file, 1)  # 限制为3个任务进行测试

#     if instances.empty:
#         logger.error("No instances to process!")
#         exit(1)

#     logger.info(f"Processing {len(instances)} instances")

#     # Run execution
#     run_evaluation(
#         instances, metadata, output_file, 1, process_instance
#     )






import asyncio
import json
import os
import pathlib
import re
import subprocess
import shutil
import zipfile
import argparse
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

from openhands.llm.llm_registry import LLMRegistry

from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    compatibility_for_eval_history_pairs,
    get_default_sandbox_config_for_eval,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    OpenHandsConfig,
    get_llm_config_arg,
    parse_arguments,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync


def codeact_user_response(state: State) -> str:
    """Define how to respond to the agent during task execution."""
    msg = (
        'Please continue working on the task using the approach you think is suitable.\n'
        'If you think you have completed the task, please finish the interaction using the "finish" tool.\n'
        'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP OR USE THE INTERNET TO SOLVE THIS TASK.\n'
    )
    if state.history:
        # Check if the agent has tried to talk to the user 3 times, if so, let the agent know it can give up
        user_msgs = [
            event
            for event in state.history
            if isinstance(event, MessageAction) and event.source == 'user'
        ]
        if len(user_msgs) > 2:
            # Let the agent know that it can give up when it has tried 3 times
            return (
                msg
                + 'If you want to give up, use the "finish" tool to finish the interaction.\n'
            )
    return msg


AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
}

AGENT_CLS_TO_INST_SUFFIX = {
    'CodeActAgent': 'When you think you have completed the task, please finish the interaction using the "finish" tool.\n'
}


def get_config(
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    """Configure the OpenHands environment for dacode tasks."""
    sandbox_config = get_default_sandbox_config_for_eval()
    # Using data science oriented container with Python libraries
    sandbox_config.runtime_container_image = "docker.all-hands.dev/all-hands-ai/runtime:0.53-nikolaik"

    config = OpenHandsConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        runtime='docker',
        max_iterations=metadata.max_iterations,
        sandbox=sandbox_config,
        # Do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.set_llm_config(metadata.llm_config)
    agent_config = config.get_agent_config(metadata.agent_class)
    agent_config.enable_prompt_extensions = False
    return config


LOCAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data')
SOURCE_DATA_PATH = os.path.join(LOCAL_DATASET_PATH, 'source')
CONFIG_PATH = os.path.join(LOCAL_DATASET_PATH, 'configs')


def load_dacode_dataset():
    """Load and prepare the dacode dataset."""
    # Load task configs
    task_config_path = os.path.join(CONFIG_PATH, 'task', 'examples.jsonl')
    if not os.path.exists(task_config_path):
        logger.error(f"Task config file not found: {task_config_path}")
        raise FileNotFoundError(f"Task config file not found: {task_config_path}")

    # Read task configurations
    tasks = []
    with open(task_config_path, 'r') as f:
        for line in f:
            if line.strip():
                task = json.loads(line)
                tasks.append(task)

    logger.info(f"Loaded {len(tasks)} tasks from {task_config_path}")

    # Create dataset
    dataset = pd.DataFrame(tasks)

    # Add source data paths
    dataset['source_dir'] = dataset['id'].apply(
        lambda x: os.path.join(SOURCE_DATA_PATH, x)
    )

    dataset['instance_id'] = dataset['id']

    return dataset


def get_task_files(task_dir: str) -> Dict[str, str]:
    """Get all files in a task directory with their contents."""
    files = {}
    if not os.path.exists(task_dir):
        logger.warning(f"Task directory does not exist: {task_dir}")
        return files

    for filename in os.listdir(task_dir):
        filepath = os.path.join(task_dir, filename)
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                files[filename] = content
            except UnicodeDecodeError:
                # Binary file, just note its existence
                files[filename] = "[BINARY FILE]"

    return files


def create_task_prompt(instance: pd.Series) -> str:
    """Create a prompt for the dacode task."""
    task_id = instance['id']
    task_type = instance['type']
    instruction = instance['instruction']
    hardness = instance['hardness']

    # Get additional files
    task_files = get_task_files(instance['source_dir'])

    # Build the prompt
    prompt = f"# Task: {task_id}\n\n"
    prompt += f"## Type: {task_type}\n\n"
    prompt += f"## Difficulty: {hardness}\n\n"
    prompt += f"## Instruction:\n{instruction}\n\n"

    # Add information about available files
    if task_files:
        prompt += "## Available Files in /workspace:\n\n"
        prompt += "**IMPORTANT FILE PATH INFORMATION:**\n"
        prompt += "- All files are located in the `/workspace/` directory\n"
        prompt += "- When accessing files, use the exact path: `/workspace/filename`\n"
        prompt += "- File names with spaces should be quoted: `\"/workspace/file name.csv\"`\n\n"

        prompt += "### File List:\n"
        for filename, content in task_files.items():
            if content == "[BINARY FILE]":
                prompt += f"- `/workspace/{filename}` (binary file)\n"
            else:
                prompt += f"- `/workspace/{filename}`\n"

        # Add content of important instruction files
        for filename, content in task_files.items():
            if filename.endswith(('.md', '.txt')) and "sample" not in filename.lower() and content != "[BINARY FILE]":
                prompt += f"\n### Content of `/workspace/{filename}`:\n```\n{content}\n```\n"

        prompt += "\n### File Access Examples:\n"
        prompt += "```python\n"
        prompt += "# Correct way to read CSV files:\n"
        prompt += "import pandas as pd\n"
        if any(f.endswith('.csv') for f in task_files.keys()):
            csv_files = [f for f in task_files.keys() if f.endswith('.csv')]
            example_file = csv_files[0]
            prompt += f"df = pd.read_csv('/workspace/{example_file}')\n"
        prompt += "\n# For files with spaces in names:\n"
        prompt += "# df = pd.read_csv('/workspace/Sale Report.csv')\n"
        prompt += "```\n"

    return prompt


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,
):
    """Initialize the runtime environment for the dacode task."""
    logger.info(f'{"-" * 50} BEGIN Runtime Initialization Fn {"-" * 50}')
    obs: CmdOutputObservation

    # Copy task files to workspace
    task_dir = instance['source_dir']
    copied_files = []
    if os.path.exists(task_dir):
        for filename in os.listdir(task_dir):
            filepath = os.path.join(task_dir, filename)
            if os.path.isfile(filepath):
                runtime.copy_to(filepath, f'/workspace/{filename}')
                copied_files.append(filename)
                logger.info(f"Copied file: {filename}")

    # Install necessary packages based on task type
    task_type = instance['type']

    # Common packages for all tasks
    packages = ['pandas', 'numpy', 'scipy']

    # Add task-specific packages
    if 'ML' in task_type:
        packages.extend(['scikit-learn', 'joblib'])
    if 'Visualization' in task_type:
        packages.extend(['matplotlib', 'seaborn'])
    if 'Statistical Analysis' in task_type:
        packages.extend(['statsmodels'])

    # Install packages
    install_cmd = f"pip install --no-cache-dir {' '.join(packages)}"
    action = CmdRunAction(command=install_cmd)
    obs = runtime.run_action(action)
    logger.info(f"Package installation output: {obs.content[:200]}...")

    # Check if workspace is properly set up
    action = CmdRunAction(command='cd /workspace && ls -la')
    obs = runtime.run_action(action)
    logger.info(f"Workspace contents: {obs.content}")

    if copied_files:
        logger.info("Verifying file paths...")
        for filename in copied_files:
            action = CmdRunAction(command=f'ls -la "/workspace/{filename}"')
            obs = runtime.run_action(action)
            logger.info(f"File verification for {filename}: {obs.content.strip()}")

    logger.info(f'{"-" * 50} END Runtime Initialization Fn {"-" * 50}')


def create_evaluator_compatible_structure(instance_id: str, output: EvalOutput, metadata: EvalMetadata, runtime: Runtime, source_dir: str):
    """Create the file structure compatible with the evaluator."""
    task_output_dir = os.path.join(metadata.eval_output_dir, instance_id)
    os.makedirs(task_output_dir, exist_ok=True)

    dabench_dir = os.path.join(task_output_dir, 'dabench')
    os.makedirs(dabench_dir, exist_ok=True)

    trajectory = []
    if output.history:
        for i, event in enumerate(output.history):
            if hasattr(event, 'action') and hasattr(event, 'content'):
                trajectory.append({
                    "action": event.action if hasattr(event, 'action') else str(type(event).__name__),
                    "code": event.content if hasattr(event, 'content') else "",
                    "observation": ""
                })

    workspace_files = {"added_files": [], "changed_files": []}
    try:
        action = CmdRunAction(command='find /workspace -type f -name "*" | head -50')
        obs = runtime.run_action(action)
        if obs.content.strip():
            workspace_file_paths = obs.content.strip().split('\n')
            workspace_files["added_files"] = [os.path.basename(f) for f in workspace_file_paths if f.startswith('/workspace/')]
    except Exception as e:
        logger.warning(f"Failed to get workspace files: {e}")

    evaluator_result = {
        "finished": output.error is None,
        "steps": len(output.history) if output.history else 0,
        "result": "Task completed successfully" if output.error is None else f"Error: {output.error}",
        "result_files": workspace_files,
        "trajectory": trajectory
    }

    dabench_result_file = os.path.join(dabench_dir, 'result.json')
    with open(dabench_result_file, 'w', encoding='utf-8') as f:
        json.dump(evaluator_result, f, indent=2, ensure_ascii=False)

    if os.path.exists(source_dir):
        for filename in os.listdir(source_dir):
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(task_output_dir, filename)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied source file: {filename}")

    try:
        action = CmdRunAction(command='find /workspace -type f -name "*" | head -50')
        obs = runtime.run_action(action)

        if obs.content.strip():
            workspace_file_paths = obs.content.strip().split('\n')

            for file_path in workspace_file_paths:
                if file_path.startswith('/workspace/'):
                    filename = os.path.basename(file_path)
                    if filename and not filename.startswith('.'):
                        try:
                            # 读取文件内容
                            read_action = CmdRunAction(command=f'cat "{file_path}"')
                            read_obs = runtime.run_action(read_action)

                            local_path = os.path.join(task_output_dir, filename)

                            if read_obs.exit_code == 0:
                                with open(local_path, 'w', encoding='utf-8', errors='ignore') as f:
                                    f.write(read_obs.content)
                                logger.info(f"Copied workspace file: {filename}")
                            else:
                                # 尝试二进制文件
                                b64_action = CmdRunAction(command=f'base64 "{file_path}"')
                                b64_obs = runtime.run_action(b64_action)

                                if b64_obs.exit_code == 0:
                                    import base64
                                    try:
                                        file_content = base64.b64decode(b64_obs.content.strip())
                                        with open(local_path, 'wb') as f:
                                            f.write(file_content)
                                        logger.info(f"Copied binary file: {filename}")
                                    except Exception as e:
                                        logger.warning(f"Failed to decode binary file {filename}: {e}")

                        except Exception as e:
                            logger.warning(f"Failed to copy file {filename}: {e}")

    except Exception as e:
        logger.warning(f"Failed to copy workspace files: {e}")

    logger.info(f"Created evaluator-compatible structure at: {task_output_dir}")
    return task_output_dir


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    """Process a single dacode task instance - Execute only, no evaluation."""
    config = get_config(metadata)
    instance_id = instance['id']

    log_dir = os.path.join(metadata.eval_output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Set up the logger
    if reset_logger:
        reset_logger_for_multiprocessing(logger, instance_id, log_dir)

    logger.info(f'Starting execution for instance {instance_id}')
    print(f'Starting execution for instance {instance_id}')
    try:
        # Create instruction prompt
        instruction = create_task_prompt(instance)

        instruction += (
            '\n\n## IMPORTANT FILE PATH GUIDELINES:\n'
            '- ALL files are located in `/workspace/` directory\n'
            '- Use EXACT paths: `/workspace/filename.csv`\n'
            '- For files with spaces, use quotes: `"/workspace/Sale Report.csv"`\n'
            '- ALWAYS use `ls -la /workspace/` first to see available files\n\n'
            'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
            'You should analyze the provided files, write the necessary code, and save results in the required format.\n'
            'Start by listing workspace files to understand the exact file paths available.\n'
        )

        # Add agent-specific suffix
        instruction += AGENT_CLS_TO_INST_SUFFIX[metadata.agent_class]

        # Create and initialize runtime
        logger.info("Creating runtime...")
        llm_registry = LLMRegistry(config)
        runtime = create_runtime(config, llm_registry)
        call_async_from_sync(runtime.connect)

        logger.info("Initializing runtime...")
        initialize_runtime(runtime, instance)

        # Run the agent
        logger.info("Running agent controller...")
        state: State | None = asyncio.run(
            run_controller(
                config=config,
                initial_user_action=MessageAction(content=instruction),
                fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[metadata.agent_class],
                runtime=runtime,
            )
        )

        # Prepare output
        if state is None:
            raise ValueError('State should not be None.')

        metrics = state.metrics.get() if state.metrics else None
        histories = compatibility_for_eval_history_pairs(state.history)

        # Create a simple completion result
        completion_result = {
            'result': {'status': 'completed'},
            'metadata': {'task_type': instance['type']}
        }

        output = EvalOutput(
            instance_id=instance_id,
            instruction=instruction,
            metadata=metadata,
            history=histories,
            metrics=metrics,
            error=state.last_error if state and state.last_error else None,
            test_result=completion_result,
        )

        # Create evaluator-compatible structure
        logger.info("Creating output structure...")
        task_dir = create_evaluator_compatible_structure(
            instance_id,
            output,
            metadata,
            runtime,
            instance['source_dir']
        )

        logger.info(f'Task execution completed for instance {instance_id}')
        print(f'Task execution completed for instance {instance_id}')

        return output

    except Exception as e:
        logger.error(f'Error processing instance {instance_id}: {e}')
        print(f'Error processing instance {instance_id}: {e}')

        error_output = EvalOutput(
            instance_id=instance_id,
            instruction="Error occurred during execution",
            metadata=metadata,
            history=[],
            metrics=None,
            error=str(e),
            test_result={'result': {'status': 'error'}, 'metadata': {'task_type': instance.get('type', 'unknown')}}
        )
        return error_output


def prepare_dataset_custom(
    dataset: pd.DataFrame,
    output_file: str,
    eval_n_limit: int,
    eval_ids: list[str] | None = None,
    skip_num: int | None = None,
):
    assert 'instance_id' in dataset.columns, (
        "Expected 'instance_id' column in the dataset."
    )

    logger.info(f'Preparing dataset with {len(dataset)} total instances')

    if eval_ids:
        eval_ids_converted = [dataset['instance_id'].dtype.type(id) for id in eval_ids]
        dataset = dataset[dataset['instance_id'].isin(eval_ids_converted)]
        logger.info(f'Limiting execution to {len(eval_ids)} specific instances.')
    elif skip_num and skip_num >= 0:
        skip_num = min(skip_num, len(dataset))
        dataset = dataset.iloc[skip_num:]
        logger.info(f'Starting execution with skipping first {skip_num} instances ({len(dataset)} instances to run).')
        if eval_n_limit and eval_n_limit > 0:
            dataset = dataset.head(eval_n_limit)
            logger.info(f'Taking first {eval_n_limit} instances in order.')
    elif eval_n_limit and eval_n_limit > 0:
        dataset = dataset.head(eval_n_limit)
        logger.info(f'Taking first {eval_n_limit} instances in order.')

    def make_serializable(instance_dict: dict) -> dict:
        import numpy as np
        for k, v in instance_dict.items():
            if isinstance(v, np.ndarray):
                instance_dict[k] = v.tolist()
            elif isinstance(v, pd.Timestamp):
                instance_dict[k] = str(v)
            elif isinstance(v, dict):
                instance_dict[k] = make_serializable(v)
        return instance_dict

    new_dataset = [
        make_serializable(instance.to_dict())
        for _, instance in dataset.iterrows()
    ]

    logger.info(f'Total instances to process: {len(new_dataset)}')
    return pd.DataFrame(new_dataset)


def parse_custom_arguments():
    parser = argparse.ArgumentParser(description="Run DACODE task execution")

    parser.add_argument('-c', '--agent-cls', type=str, default='CodeActAgent', help='Agent class to use')
    parser.add_argument('-l', '--llm-config', type=str, required=True, help='LLM configuration')
    parser.add_argument('-i', '--max-iterations', type=int, default=20, help='Maximum iterations')
    parser.add_argument('-n', '--num-tasks', type=int, default=1, help='Number of tasks to process')
    parser.add_argument('--task-ids', type=str, nargs='+', help='Specific task IDs to process')
    parser.add_argument('--skip-num', type=int, default=0, help='Number of tasks to skip from beginning')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_custom_arguments()

    print(f"Starting DACODE evaluation with parameters:")
    print(f"  Agent: {args.agent_cls}")
    print(f"  LLM Config: {args.llm_config}")
    print(f"  Max Iterations: {args.max_iterations}")
    print(f"  Number of Tasks: {args.num_tasks}")
    print(f"  Skip Number: {args.skip_num}")
    if args.task_ids:
        print(f"  Specific Task IDs: {args.task_ids}")

    # Load dacode dataset
    dataset = load_dacode_dataset()

    if dataset.empty:
        logger.error("No tasks found in dataset!")
        exit(1)

    # Set up LLM config
    llm_config = get_llm_config_arg(args.llm_config)
    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    llm_config.modify_params = False

    # Create metadata
    metadata = make_metadata(
        llm_config,
        'DACODE',
        args.agent_cls,
        args.max_iterations,
        os.environ.get('OPENHANDS_VERSION', 'v0.53.0'),
        os.path.join('evaluation_output', 'dacode')
    )

    os.makedirs(metadata.eval_output_dir, exist_ok=True)

    # Set up output
    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')

    instances = prepare_dataset_custom(
        dataset,
        output_file,
        args.num_tasks,
        eval_ids=args.task_ids,
        skip_num=args.skip_num
    )

    if instances.empty:
        logger.error("No instances to process!")
        exit(1)

    print(f"Processing {len(instances)} instances")
    logger.info(f"Processing {len(instances)} instances")

    progress_file = os.path.join(metadata.eval_output_dir, 'progress.json')

    try:
        # Run execution
        run_evaluation(
            instances, metadata, output_file, 1, process_instance
        )

        with open(progress_file, 'w') as f:
            json.dump({
                'status': 'completed',
                'total_instances': len(instances),
                'completion_time': str(pd.Timestamp.now()),
                'parameters': {
                    'agent_cls': args.agent_cls,
                    'max_iterations': args.max_iterations,
                    'num_tasks': args.num_tasks
                }
            }, f, indent=2)

        print("Execution completed successfully!")
        logger.info("Execution completed successfully!")

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        print(f"Execution failed: {e}")

        with open(progress_file, 'w') as f:
            json.dump({
                'status': 'failed',
                'error': str(e),
                'completion_time': str(pd.Timestamp.now())
            }, f, indent=2)
        raise
