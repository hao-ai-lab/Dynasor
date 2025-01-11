from collections import Counter
from openai import OpenAI
import json
import sys
import random
import time
from concurrent.futures import ThreadPoolExecutor
import wikienv, wrappers


# Initialize OpenAI client with vLLM's API server
client = OpenAI(
    api_key="EMPTY", 
    base_url="http://localhost:30000/v1"
)


# Initialize Dataset
env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

def get_react_response(model: str, prompt: str, stop=["\n"]):
    """
    Generate multiple completions for a prompt and determine the most consistent result.

    Parameters:
    - model (str): The model name.
    - prompt (list): The input prompt.
    - stop (list): The stop tokens. 

    Returns:
    - str: The string of completions.
    """
    response = client.completions.create(
      model="meta-llama/Meta-Llama-3-8B-Instruct",
      prompt=prompt,
      temperature=0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=stop
    )
    
    return response.choices[0].text


def prepare_prompt_react() -> str:
    """
    Prepare the prompt for the model.
    
    Parameters:
    - question (str): The input question to be answered
    
    Returns:
    - str: The formatted prompt
    """

    folder = './react_prompts/'
    prompt_file = 'prompts_naive.json'
    with open(folder + prompt_file, 'r') as f:
        prompt_dict = json.load(f)
    webthink_examples = prompt_dict['webthink_simple6']
    
    instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
    prompt = instruction + webthink_examples
    
    return prompt

def webthink(idx=None, to_print=True, prompt=""):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1
        next_prompt = prompt + f"Thought {i}:"
        thought_action = get_react_response(model="", prompt=next_prompt, stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            next_round = prompt + f"Thought {i}: {thought}\nAction {i}:"
            action = get_react_response(model="", prompt=next_round, stop=[f"\n"]).strip()
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        if to_print:
            print(step_str)
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")
    if to_print:
        print(info, '\n')
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info

def test_react_hotpotqa(num_q: int):
    idxs = list(range(7405))
    random.Random(233).shuffle(idxs)

    rs = []
    infos = []
    old_time = time.time()

    prompt = prepare_prompt_react()
    for i in idxs[:num_q]:
        r, info = webthink(i, to_print=True, prompt=prompt)
        rs.append(info['em'])
        infos.append(info)
        print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
        print('-----------')
        print()

if __name__ == "__main__":
    test_react_hotpotqa(500)

