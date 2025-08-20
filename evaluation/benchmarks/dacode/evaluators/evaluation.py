import logging
import os, json
from typing import Callable, Any
from typing import List, Dict, Union
from pathlib import Path
import sys, jsonlines
import signal
from contextlib import contextmanager
here=Path(__file__).absolute()
sys.path.append(str(here.parent))
import metrics
from tqdm import tqdm
import re
import traceback

Metric = Callable[[Any, Any], float]

@contextmanager
def timeout(seconds, error_message="Timeout"):
    """Simple timeout context manager"""
    def timeout_handler(signum, frame):
        raise TimeoutError(error_message)

    # Set the signal handler and a alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the old signal handler and cancel the alarm
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)

class Evaluator:

    def __init__(self, output_dir: str, gold_dir: str, timeout_seconds: int = 10):
        self.output_dir = output_dir
        self.gold_dir = gold_dir
        self.timeout_second = timeout_seconds

    def get_result_file(self, results: List, dir: str, isgold: bool):
        results = results if isinstance(results, list)\
            else [results]
        if 'number' in results[0].keys():
            return 'number', [results[0]['number']]
        result_files = []
        for result in results:
            multi = result.get("multi", False)
            files = result['file'] if isinstance(result['file'], list) \
                else [result['file']]
            if multi:
                files = [os.path.join(dir, file) for file in files] if not isgold \
                    else [os.path.join(dir, os.path.basename(file)) for file in files]
                result_files.append(files)
            else:
                for file in files:
                    file = file if not isgold else os.path.basename(file)
                    result_files.append(os.path.join(dir, file))
        return 'file', result_files

    def _get_eval_config_info(self, eval_config: Dict[str, Any]):
        id = eval_config['id']
        output_id_dir = os.path.join(self.output_dir, id)

        result_file = os.path.join(output_id_dir, 'dabench', 'result.json')


        if not os.path.exists(result_file):
            print(f"Result file not found: {result_file}")
            trajectory_info = {
                "finished": False,
                "steps": 0,
                "result": "No result file found",
                "added_files": [],
                "changed_files": [],
                "actions": [],
                "hardness": eval_config.get('config', {}).get('hardness', "none")
            }
        else:
            trajectory_info = self._get_trajectory_info_from_json(result_file)

        gold_id_dir = os.path.join(self.gold_dir, id)
        config = eval_config.get('config', {})
        hardness = config.get('hardness', "none")
        trajectory_info = {"hardness": hardness, **trajectory_info}
        metric_conj: str = eval_config.get("conj", "avg")
        expected = eval_config['result'] if isinstance(eval_config['result'], list) \
            else [eval_config['result']]

        type, gold_results = self.get_result_file(expected, dir=gold_id_dir, isgold=True)
        if type == 'number':
            output_type = ['.txt', '.json', '.png', '.jpg', '.csv', '.npy']
            output = trajectory_info["result"]
            is_file = any(map(lambda x: output.endswith(x), output_type))
            if not is_file:
                output_results = [trajectory_info["result"]]
            else:
                output_results = self._get_result_file_from_json(output_id_dir, trajectory_info["result"], is_plot=(config.get("task") == "data visualization"))
        else:
            output_results = self._get_result_file_from_json(output_id_dir, trajectory_info["result"], is_plot=(config.get("task") == "data visualization"))
            if len(output_results) != len(gold_results):
                _, output_results = self.get_result_file(expected, dir=output_id_dir, isgold=False)

        metric: Metric = [getattr(metrics, func) for func in eval_config["func"]] \
            if isinstance(eval_config["func"], list)\
            else [getattr(metrics, eval_config["func"])]
        metric_options: Union[List[Dict[str, Any]], Dict[str, Any]] = \
            [opt if opt else {} for opt in eval_config["options"]] \
            if isinstance(eval_config.get("options", {}), list) \
            else eval_config["options"] \
            if "options" in eval_config.keys() \
            else [{}] * len(metric) \
            if isinstance(metric, list) \
            else {}

        metric = metric * len(output_results) if len(output_results) > len(metric) and len(metric) == 1  \
            else metric
        metric_options = metric_options * len(output_results) if len(output_results) > len(metric_options) \
            and len(metric_options) == 1  \
            else metric_options

        assert (not isinstance(eval_config["func"], list)
            or (len(metric) == len(output_results) == len(gold_results) == len(
                metric_options))), "Evaluation configs need to be consistent: lengths of 'metric', 'output_results', 'gold_results', " \
            "and 'metric_options' must be the same when 'func' is a list."

        return id, True, trajectory_info, (config, metric, metric_conj, metric_options, output_results, gold_results)

    def _get_trajectory_info_from_json(self, result_file):
        with open(result_file, 'r') as f:
            result = json.load(f)

        trajectory = result.get("trajectory", [])
        actions = []

        for i, step in enumerate(trajectory):
            if i+1 < len(trajectory):
                observation = trajectory[i+1].get("observation", "")
                if "executed successfully. No output." in observation:
                    observation = "execution succeeded"
                elif observation.startswith("Failed to parse action from your response,"):
                    observation = "action parse failed"
                elif observation.startswith("ERROR:") or "Traceback (" in observation or \
                    observation.startswith("bash: -c: line ") or "Error: " in observation:
                    observation = "error message"
                elif "Warning:" in observation:
                    observation = "warning message"
                else:
                    observation = "standard output"
            else:
                observation = ""

            action = step.get("action", "")
            code = step.get("code", "")

            if action.startswith("Bash"):
                actions.append(("Bash", code, observation))
            elif action.startswith("Python"):
                actions.append(("Python", len(code.split("\n")), observation))
            elif action.startswith("SQL"):
                actions.append(("SQL", code, observation))
            elif action.startswith("Terminate"):
                actions.append(("Terminate", "", observation))
            else:
                actions.append(("None", "", observation))

        info = {
            "finished": result.get("finished", False),
            "steps": result.get("steps", 0),
            "result": result.get("result", ""),
            "added_files": result.get("result_files", {}).get("added_files", []),
            "changed_files": result.get("result_files", {}).get("changed_files", []),
            "actions": actions
        }
        return info

    def _get_result_file_from_json(self, output_id_dir, result_file, is_plot=False):
        pattern = r'\b(?:[\w/\-_]+/)?([\w\-_]+(\.\w+)+)\b'
        filenames = re.findall(pattern, result_file)
        if not filenames:
            return []
        filenames = [filename[0] for filename in filenames]
        result_file = [os.path.join(output_id_dir, file) for file in filenames]

        if is_plot:
            result_file += [os.path.join(output_id_dir,"dabench/plot.json"), os.path.join(output_id_dir,"dabench/result.npy")]
            result_file = [result_file]
        return result_file

    def evaluate(self, env_config: Union[Dict, str, List]):
        """
        Evaluate task
        """
        if isinstance(env_config, str):
            if not os.path.exists(env_config) or not os.path.isfile(env_config):
                raise ValueError('File Path Error: Please provide a right file path')
            if env_config.endswith('.json'):
                with open(env_config, 'r') as f:
                    env_configs = json.load(f)
            elif env_config.endswith('.jsonl'):
                with jsonlines.open(env_config, 'r') as js:
                    env_configs = [config_eval for config_eval in js]
            else:
                raise ValueError('File Type Error: Please Upload json or jsonl file')
            env_configs = env_configs if isinstance(env_configs, list) else [env_configs]
        elif isinstance(env_config, dict):
            env_configs = [env_config]
        elif isinstance(env_config, list):
            env_configs = env_config
        else:
            raise ValueError('Invalid env_config type')

        eval_results = []
        pbar = tqdm(total=len(env_configs))

        for eval_config in env_configs:
            id = eval_config.get('id', 'unknown')
            pbar.set_description(f"Processing Task id: {id}")
            pbar.update(1)

            try:
                id, exist, trajectory_info, eval_info = self._get_eval_config_info(eval_config)

                if not exist:
                    print(f"Result of Task {id} does not exist!")
                    continue

                (config, metric_list, metric_conj, metric_options, output_results, gold_results) = eval_info
                task_type = config.get('task', 'unknown')
                hardness = config.get('hardness', 'unknown')
                result_type = config.get('type', 'unknown')

                if trajectory_info["finished"] == False:
                    eval_results.append({"id": id, "task":task_type,"result_type":result_type, "hardness":hardness, "total_score": 0.0, **trajectory_info})
                    continue

                if metric_list == "infeasible":
                    eval_results.append({"id": id, "task":task_type,"result_type":result_type, "hardness":hardness, "total_score": 0.0, **trajectory_info})
                    continue

                try:
                    with timeout(self.timeout_second, "Action execution time exceeded!"):
                        scores = []
                        info = []

                        for idx, metric in enumerate(metric_list):
                            try:
                                output_result = output_results[idx]
                                gold_result = gold_results[idx]
                                if config:
                                    config_copy = {"config": config}
                                    metric_options[idx].update(config_copy)

                                result = metric(output_result, gold_result,**metric_options[idx])
                            except FileNotFoundError as e:
                                logging.error(f"File not found! Error: {e}")
                                scores.append(0.0)
                                continue
                            except Exception as e:
                                logging.error(f"Error in metric evaluation: {e}")
                                scores.append(0.0)
                                continue

                            if isinstance(result, dict):
                                scores.append(result.get('score', 0.0))
                                output_result = output_result if isinstance(output_result, list) else [output_result]
                                result['file'] = [os.path.basename(file) for file in output_result]
                                info.append(result)
                            else:
                                scores.append(result)

                except TimeoutError as e:
                    logging.error(f"Timeout in task {id}: {e}")
                    scores = [0.0]
                    info = [{"score": 0.0, "errors": [str(e)], 'file': []}]
                except Exception as e:
                    logging.error(f"Error in task {id}: {e}")
                    traceback.print_exc()
                    scores = [0.0]
                    info = [{"score": 0.0, "errors": [str(e)], 'file': []}]

                if metric_conj == 'avg':
                    total_score = sum(scores) / len(scores) if scores else 0.0
                elif metric_conj == 'max':
                    total_score = max(scores) if scores else 0.0
                elif metric_conj == 'min':
                    total_score = min(scores) if scores else 0.0
                elif metric_conj == 'and':
                    total_score = float(all(score!= 0 for score in scores)) if scores else 0.0
                elif metric_conj == 'or':
                    total_score = float(any(score!= 0 for score in scores)) if scores else 0.0
                else:
                    total_score = 0.0

                eval_results.append({"id": id, "task":task_type,"result_type":result_type, "hardness":hardness, "total_score": total_score, **trajectory_info, 'info': info})

            except Exception as e:
                logging.error(f"Error processing task {id}: {e}")
                traceback.print_exc()
                eval_results.append({"id": id, "task":"unknown","result_type":"unknown", "hardness":"unknown", "total_score": 0.0, "finished": False, "steps": 0, "result": str(e), "added_files": [], "changed_files": [], "actions": [], 'info': []})

        pbar.close()
        return eval_results
