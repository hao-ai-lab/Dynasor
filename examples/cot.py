from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from gsm8k_utils import extract_answer, math_equal, load_jsonl
import os

# Initialize OpenAI client with vLLM's API server
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_api_base = os.environ.get("OPENAI_API_BASE")
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# TODO(GindaChen) Add logprob and use certaindex to control stopping
def get_cot_response(model: str, messages: list, sampling_params: dict = None):
    """
    Generate a single chain-of-thought completion for a prompt.

    Parameters:
    - model (str): The model name.
    - messages (list): The input messages.
    - sampling_params (dict): Parameters for sampling.

    Returns:
    - str: The completion text.
    """
    sampling_params = sampling_params or dict(
        max_tokens=512,
        temperature=0.7,
    )
    print('model: ', model)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        **sampling_params,
    )
    return completion.choices[0].message.content

def prepare_prompt_gsm8k(question: str) -> str:
    """
    Prepare the prompt for the model with system and user messages.
    
    Parameters:
    - question (str): The input question to be answered
    
    Returns:
    - str: The formatted prompt with system and user messages
    """
    system_msg = "You are a helpful assistant."
    user_msg = f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    return messages

def load_gsm8k_questions():
    return load_jsonl("data/GSM8K/test.jsonl")


def test_one_cot_gsm8k():
    model_name = "deepseek-chat"
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    messages = prepare_prompt_gsm8k(question)
    sampling_params = dict(
        temperature=0.7,
        max_tokens=512,
    )
    response = get_cot_response(
        model=model_name,
        messages=messages,
        sampling_params=sampling_params,
    )
    answer = extract_answer(response)
    return answer

def cot_gsm8k_process_item(item, model_name, sampling_params):
    question = item["question"]
    _answer = item["answer"]
    answer = extract_answer(_answer)
    messages = prepare_prompt_gsm8k(question)
    response = get_cot_response(
        model=model_name,
        messages=messages,
        sampling_params=sampling_params,
    )
    final_answer = extract_answer(response)
    return dict(
        is_equal=math_equal(answer, final_answer),
        ground_truth_answer=answer,
        final_answer=final_answer,
        # TODO(GindaChen) Add reasoning field?
    )

def test_cot_gsm8k(
    is_parallel: bool = False,
    n_dataset_rows: int = 5,
):
    # Load test dataset
    gsm8k_test_dataset = load_gsm8k_questions()
    gsm8k_test_dataset = gsm8k_test_dataset[:n_dataset_rows]

    model_name = "deepseek-chat"
    sampling_params = dict(
        temperature=0.7,  # Enable some variability in responses
        max_tokens=512,
    )
    
    if is_parallel:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for item in gsm8k_test_dataset:
                future = executor.submit(
                    cot_gsm8k_process_item,
                    item,
                    model_name,
                    sampling_params,
                )
                futures.append(future)
            results = [f.result() for f in futures]
    else:
        results = []
        for item in gsm8k_test_dataset:
            result = cot_gsm8k_process_item(
                item,
                model_name,
                sampling_params,
            )
            print(result)
            results.append(result)

    correct_count = 0
    for result in results:
        if result["is_equal"]:
            correct_count += 1
    
    accuracy = correct_count / len(gsm8k_test_dataset)
    print(f"Accuracy: {correct_count} / {len(gsm8k_test_dataset)} = {accuracy * 100:.2f}%")
    return dict(
        accuracy=accuracy,
        results=results,
    )



if __name__ == "__main__":
    # test_one_cot_gsm8k()
    # test_cot_gsm8k(is_parallel=False)
    test_cot_gsm8k(is_parallel=True)

