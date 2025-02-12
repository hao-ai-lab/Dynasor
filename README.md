
# Dynasor

## This is a simple tool to help you speed up LLM reasoning (e.g., Long Chain of Thought).

### How it works

Dynasor uses a combination of techniques to speed up LLM reasoning:

1. **Prompt Engineering**: We use a combination of techniques to improve the prompt.
2. **Dynamic Execution**: We dynamically execute the prompt, and stop when the LLM has enough information to make a decision.



# Launch Server

waring: note to enable prefix caching, or probing will be very slow.
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -tp 1  --api-key token-abc123 --enable-prefix-caching
```
# Launch Client 
```bash
# Enable Dynasor
python chat_complete.py --dynasor-saving-effort crazy
# Disable Dynasor
python chat_complete.py 
```

# Token Deprivation Experiment

```bash
python run.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --dataset math500 --step 32 --max-tokens 16384 --start 0 --end 10 --output ./math500_step32_max16384_trials10  --probe-tokens 32 --probe "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{" 

```

