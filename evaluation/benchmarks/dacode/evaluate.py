import argparse
import json
import os
import pandas as pd
from pathlib import Path
import sys

eval_dir = Path(__file__).parent / "evaluators"
sys.path.append(str(eval_dir))

from evaluation import Evaluator

def get_available_tasks(execution_dir, eval_config_path):
    with open(eval_config_path, 'r') as f:
        eval_configs = [json.loads(line) for line in f if line.strip()]

    available_configs = []
    for config in eval_configs:
        task_id = config['id']
        task_dir = os.path.join(execution_dir, task_id)
        dabench_result = os.path.join(task_dir, "dabench", "result.json")

        if os.path.exists(task_dir) and os.path.exists(dabench_result):
            available_configs.append(config)
            print(f"Found task: {task_id}")
        else:
            print(f"Skipping task {task_id}: directory or result file not found")

    return available_configs

def run_evaluation(execution_dir, gold_dir, eval_config_path, result_dir, timeout_seconds=300):
    print(f"Starting evaluation...")
    print(f"Execution dir: {execution_dir}")
    print(f"Gold dir: {gold_dir}")
    print(f"Eval config: {eval_config_path}")

    print("\nChecking available tasks...")
    available_configs = get_available_tasks(execution_dir, eval_config_path)

    if not available_configs:
        print("No valid tasks found for evaluation!")
        return

    print(f"\nFound {len(available_configs)} tasks to evaluate")

    evaluator = Evaluator(output_dir=execution_dir, gold_dir=gold_dir, timeout_seconds=timeout_seconds)

    print("Running evaluation...")
    results_infos = []

    for config in available_configs:
        try:
            result = evaluator.evaluate([config])
            if result:
                results_infos.extend(result)
        except Exception as e:
            print(f"Error evaluating task {config['id']}: {e}")
            failed_result = {
                'id': config['id'],
                'task': config['config'].get('task', 'unknown'),
                'result_type': config['config'].get('type', 'unknown'),
                'hardness': config['config'].get('hardness', 'unknown'),
                'total_score': 0.0,
                'finished': False
            }
            results_infos.append(failed_result)

    if not results_infos:
        print("No evaluation results generated!")
        return

    num_results = len(results_infos)
    scores = [result.get('total_score', 0.0) for result in results_infos]
    finished = [result.get('finished', False) for result in results_infos]
    result_types = [result.get('result_type', 'unknown') for result in results_infos]
    task_types = [result.get('task', 'unknown') for result in results_infos]
    hardness_levels = [result.get('hardness', 'unknown') for result in results_infos]

    task_types = ["machine learning" if "machine learning" in t else t for t in task_types]
    eda_types = ["data insight", "data manipulation", "data visualization", "statistical analysis"]
    big_types = ["EDA" if t in eda_types else t for t in task_types]
    plot_types = ["line", "pie", "bar", "scatter"]
    result_types = ["plot" if t in plot_types else t for t in result_types]

    df = pd.DataFrame({
        "type": task_types,
        "score": scores,
        "finished": finished,
        "hardness": hardness_levels,
        "big_type": big_types,
        "result_type": result_types
    })

    average_score = sum(scores) / num_results if num_results > 0 else 0.0
    average_finished = sum(finished) / num_results if num_results > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Number of results: {num_results}")
    print(f"Average score: {average_score:.3f}")
    print(f"Average finished: {average_finished:.3f}")

    print(f"\n{'='*50}")
    print("By Task Type:")
    print(df.groupby("type").agg({"score": "mean", "finished": "mean"}))

    print(f"\n{'-'*30}")
    print("By Hardness:")
    print(df.groupby("hardness").agg({"score": "mean", "finished": "mean"}))

    print(f"\n{'-'*30}")
    print("By Big Type:")
    print(df.groupby("big_type").agg({"score": "mean", "finished": "mean"}))

    print(f"\n{'-'*30}")
    print("By Result Type:")
    print(df.groupby("result_type").agg({"score": "mean", "finished": "mean"}))

    os.makedirs(result_dir, exist_ok=True)
    results_json = {
        "num_results": num_results,
        "average_score": average_score,
        "average_finished": average_finished,
        "results": results_infos,
        "statistics": {
            "by_type": df.groupby("type").agg({"score": "mean", "finished": "mean"}).to_dict(),
            "by_hardness": df.groupby("hardness").agg({"score": "mean", "finished": "mean"}).to_dict(),
            "by_big_type": df.groupby("big_type").agg({"score": "mean", "finished": "mean"}).to_dict(),
            "by_result_type": df.groupby("result_type").agg({"score": "mean", "finished": "mean"}).to_dict()
        }
    }

    output_file = os.path.join(result_dir, f"{os.path.basename(execution_dir)}_evaluation.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluator for DACODE tasks")
    parser.add_argument(
        "--execution_dir",
        type=str,
        required=True,
        help="Directory containing task execution results"
    )
    parser.add_argument(
        "--gold_dir",
        type=str,
        default="evaluation/benchmarks/dacode/data/gold",
        help="Directory containing gold standard files"
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        default="evaluation/benchmarks/dacode/data/configs/eval/eval_all.jsonl",
        help="Evaluation configuration file"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for each evaluation in seconds"
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    if not os.path.exists(args.execution_dir):
        print(f"Error: Execution directory not found: {args.execution_dir}")
        return

    if not os.path.exists(args.gold_dir):
        print(f"Error: Gold directory not found: {args.gold_dir}")
        return

    if not os.path.exists(args.eval_config):
        print(f"Error: Evaluation config file not found: {args.eval_config}")
        return

    run_evaluation(
        execution_dir=args.execution_dir,
        gold_dir=args.gold_dir,
        eval_config_path=args.eval_config,
        result_dir=args.result_dir,
        timeout_seconds=args.timeout
    )

if __name__ == "__main__":
    main()
