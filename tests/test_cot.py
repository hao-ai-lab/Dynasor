from dynasor import DeepSeekClient 
from dynasor.algorithms import CoT
from tqdm import tqdm
from dynasor.datasets import MathDatasetLoader
from dynasor.evaluate import MathEvaluator
from dynasor.evaluate.math_evaluator import majority_vote, extract_answer
import os 

def test_cot_gsm8k(is_parallel=True, n_dataset_rows=5):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_api_base = os.environ.get("OPENAI_API_BASE")

    client = DeepSeekClient(api_key=openai_api_key, api_base=openai_api_base)
    CoTInstance = CoT(client)
    dataset = MathDatasetLoader(dataset_path="../data/GSM8K/test.jsonl")
    evaluator = MathEvaluator(voting='majority')

    def prompt_func(item: dict) -> str:
        question = item["question"]
        system_msg = "You are a helpful assistant."
        user_msg = f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        return messages

    def ground_truth_func(item: dict) -> dict:
        return extract_answer(item["answer"], 'gsm8k')

    dataset.load(n_rows=1)
    messages = dataset.prepare_dataset(prompt_func)
    ground_truths = dataset.prepare_dataset(ground_truth_func)

    print(len(messages))
    results = []
    for message in tqdm(messages):
        result = CoTInstance.run(message)
        results.append(result)

    metric = evaluator.evaluate(results, ground_truths, extract_func=extract_answer)
    print('Accuracy: ', metric)
    # results = CoTInstance.run_batch(
    #     client, messages, is_parallel=is_parallel,
    # )
    return results


if __name__ == "__main__":
    results = test_cot_gsm8k(is_parallel=True, n_dataset_rows=1)

