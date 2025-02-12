import argparse
import random
import os
from tqdm import tqdm
from dynasor.utils import load_jsonl, save_json, load_dataset, set_seed
from dynasor.evaluator import extract_answer, strip_string, math_equal, extract_first_boxed_answer
from clients import vllmClientModel


def parse_args():
    parser = argparse.ArgumentParser(description='Token Deprivation Experiment')
    parser.add_argument('--dataset', type=str, required=True, choices=['amc23', 'aime24', 'GPQADiamond', 'math500'],
                      help='Dataset to use (amc23 or aime24 or math500)')
    parser.add_argument('--output', type=str, default="",
                      help='Path to output results file')
    parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
                      help='Name or path of the model to use')
    parser.add_argument('--probe', type=str, default="**Final Answer**\n\n\\[ \\boxed{",
                      help='probe the LLM to output the answer in the format of boxed{...}')
    parser.add_argument('--probe-tokens', type=int, default=10,
                      help='Number of tokens in probe')

    parser.add_argument('--url', type=str, default='http://localhost:8000/v1',
                      help='URL of the model to use')
    parser.add_argument('--api-key', type=str, default='token-abc123',
                      help='API key of the model to use')
    parser.add_argument('--start', type=int, default=0,
                      help='Start index of the question')
    parser.add_argument('--end', type=int, default=10000,
                      help='End index of the question')
    parser.add_argument('--max-tokens', type=int, default=8192,
                      help='Maximum number of tokens per request')
    parser.add_argument('--step', type=int, default=128,
                      help='Step size for token budget')
    parser.add_argument('--num-trials', type=int, default=10,
                      help='Number of trials per question')

    parser.add_argument('--temperature', type=float, default=0.6,
                      help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.95,
                      help='Top p for sampling')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
        
    return parser.parse_args()



def load_model(model_name, url, api_key):

    return vllmClientModel(model_name, url, api_key)


def execute_question_reuse(model, prompt, target, max_tokens=[2048], probe=None, probe_tokens=10, num_trials=10, problem_id=None, output_dir=None, top_p=0.95, temperature=0.6):
    results = []
    current_prompts = [model.prepare_prompt(prompt) for _ in range(num_trials)]
    for i in tqdm(range(len(max_tokens)), desc="Executing questions"):
        #print(f"Executing question {i} with max tokens {max_tokens[i]}")
        
        # Track which trials are finished
        if i == 0:
            is_finished = [False] * num_trials
            responses = model.generate_batch(current_prompts, max_tokens=max_tokens[i], is_actives=[True] * num_trials, top_p=top_p, temperature=temperature)
        else:
            # Calculate remaining tokens needed
            remaining_tokens = max_tokens[i] - max_tokens[i-1]
            # Stitch previous response to prompt
            current_prompts = [
                current_prompt + completion[0]
                for current_prompt, completion in zip(current_prompts, completions)
            ]
            # Only generate for unfinished trials
            responses = model.generate_batch(current_prompts, max_tokens=remaining_tokens, is_actives=[not finished for finished in is_finished], top_p=top_p, temperature=temperature)

        #print(responses)
        completions = []
        for trial in range(num_trials):
            if is_finished[trial]:
                completions.append(("", None))  # Empty completion for finished trials
            else:
                response = responses[trial]
                if response is None:
                    completions.append(("", None))
                else:
                    text = response.choices[0].text
                    finish_reason = response.choices[0].finish_reason
                    logprobs = response.choices[0].logprobs
                    completions.append((text, finish_reason))
                    # Update finished status if LLM completed naturally
                    if finish_reason != 'length':
                        is_finished[trial] = True

        # Save results for this round
        round_results = {
            'round': i,
            'problem_id': problem_id,
            'max_tokens': max_tokens[i],
            'prompts': current_prompts,
            'new_tokens': [completion[0] for completion in completions],
            'finish_reasons': [completion[1] for completion in completions],
            'is_finished': is_finished,
            "target" : target
        }

        # Generate and save probed responses
        probe_prompts = [
            current_prompt + completion[0] + probe 
            for current_prompt, completion in zip(current_prompts, completions)
        ]
        # Only generate probe responses for unfinished trials
        probe_responses = model.generate_batch_probe(probe_prompts, max_tokens=probe_tokens, is_actives=[not finished for finished in is_finished])
        
        round_results['probe_prompts'] = probe_prompts
        round_results['probe_responses'] = [
            response.choices[0].text if response else ""
            for response in probe_responses
        ]

        is_corrects = []
        is_corrects_original = []
        for trial in range(num_trials):
            if is_finished[trial]:
                finished_result = extract_answer(current_prompts[trial] + completions[trial][0], 'aime24')
                #print('Result Corrects: ', finished_result, target, math_equal(finished_result, target))
                is_corrects.append(math_equal(finished_result, target))
            else:
                probe_result = extract_first_boxed_answer(probe_prompts[trial] + probe_responses[trial].choices[0].text, 'aime24',)
                #print('Probe Corrects: ', probe_result, target, math_equal(probe_result, target))
                is_corrects.append(math_equal(probe_result, target))
            
            is_corrects_original.append(math_equal(extract_answer(current_prompts[trial] + completions[trial][0], 'aime24'), target))
                
        round_results['is_corrects'] = is_corrects
        round_results['is_corrects_original'] = is_corrects_original

        # Save results for this round to a file
        if output_dir:
            round_filename = f"{output_dir}/question_{problem_id}_tokens_{max_tokens[i]}.json"
            save_json(round_results, round_filename)
        
def main():
    args = parse_args()
    set_seed(args.seed)
    data = load_dataset(args.dataset)
    
    num_trials = args.num_trials  # Number of trials per question
    
    import os
    from datetime import datetime
    import random
    import string
    
    if args.output:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Create output directory with model name, dataset, parameters and date
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        model_name = args.model.replace('/', '-')
        output_dir = f"results/{model_name}_{args.dataset}_step{args.step}_max{args.max_tokens}_trials{args.num_trials}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
    model = load_model(args.model, args.url, args.api_key)

    for problem_id, item in enumerate(data):
        if problem_id < args.start:
            continue
        if problem_id >= args.end:
            break
        # execute question
        prompt = item['problem'].strip()
        target = strip_string(item['answer'])

        print(f"Executing question {problem_id} with target [{target}]")
        print(f"Prompt: {prompt}")
        print("-"*100)
        token_budgets = list(range(args.step, args.max_tokens + args.step, args.step))
        batch_results = execute_question_reuse(model, prompt, target, max_tokens=token_budgets, probe=args.probe, probe_tokens=args.probe_tokens,num_trials=num_trials, problem_id=problem_id, output_dir=output_dir, top_p=args.top_p, temperature=args.temperature)

    print('Saved results to', output_dir)

if __name__ == '__main__':
    main()