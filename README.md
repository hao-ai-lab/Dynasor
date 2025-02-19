
<div align="center">
    <h1>Dynasor🦖: Efficiently Serving LLM Reasoning Programs with Certaindex</h1>
</div>

<!-- https://hao-ai-lab.github.io/blogs/dynasor-cot/ -->
<!-- <div align="center" style="line-height: 1;">
    <a href="https://viol2000.github.io/SubDomain/gradio-2.html" target="_blank" style="margin: 2px;">
        <img alt="Demo" src="https://img.shields.io/badge/🤖Chat-Deepseek-blue?" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://x.com/haoailab" target="_blank" style="margin: 2px;">
        <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-haoailab-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://hao-ai-lab.github.io/blogs/dynasor-cot/" target="_blank" style="margin: 2px;">
        <img alt="Blog" src="https://img.shields.io/badge/Blog-Dynasor-4CAF50?&color=4CAF50" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://arxiv.org/abs/2412.20993" style="margin: 2px;">
        <img alt="License" src="https://img.shields.io/badge/Paper-Dynasor-4CAF50?&color=4CAF50" style="display: inline-block; vertical-align: middle;"/>
    </a>
</div> -->

<div align="center" style="line-height: 1; margin-bottom: 12px;">
    | <a href="https://hao-ai-lab.github.io/blogs/dynasor-cot/">📝 Blog</a> 
    | <a href="https://arxiv.org/abs/2412.20993">📄 Paper</a> 
    | <a href="https://e4d417385887b7e801.gradio.live/">🤖 Demo</a> 
    | <a href="https://x.com/haoailab">🐦 Twitter/X</a> 
    |
</div>


Simple extension on vLLM to help you speed up reasoning model without training.


- [Quick Start](#quick-start)
- [What is Dynasor](#what-is-dynasor)
- [How to use Dynasor](#how-to-use-dynasor)
  - [dynasor-chat: CLI Chat Interface](#dynasor-chat-cli-chat-interface)
  - [dynasor-openai: OpenAI Compatible Server](#dynasor-openai-openai-compatible-server)
  - [dynasor-vllm: vLLM Native Server](#dynasor-vllm-vllm-native-server)

---

Try our <a href="https://e4d417385887b7e801.gradio.live/">🤖 Demo</a>!

<div align="center">
    <img src="assets/demo.gif" alt="Dynasor Demo" width="600px">
</div>



## Quick Start 

- [Run Dynasor locally](./docs/local.md)


Use Dynasor:
```bash
# Install Dynasor
git clone https://github.com/hao-ai-lab/Dynasor.git 
cd Dynasor && pip install . && cd -

# (Optional) Install and setup vllm endpoint
pip install vllm
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -tp 1 --enable-prefix-caching

# Start Dynasor Chat with an endpoint
dynasor-chat --base-url http://localhost:8000/v1
```


## What is Dynasor?

Dynasor is a tool that helps you speed up LLM reasoning model without training or finetuning. It uses a combination of techniques to improve the prompt, and dynamically execute the prompt, and stop when the LLM has enough information to make a decision. 



## Installation


### Install via source
```bash
git clone https://github.com/hao-ai-lab/Dynasor.git
cd Dynasor && pip install . && cd -
```


## How to use Dynasor

We provide 3 tools to launch Dynasor:

1. [`dynasor-chat`](#dynasor-chat-cli-chat-interface): CLI chat interface to interact with Dynasor
2. [`dynasor-openai`](#dynasor-openai-openai-compatible-server): OpenAI compatible server.
3. [`dynasor-vllm`](#dynasor-vllm-vllm-native-server): vLLM-native server





### `dynasor-chat`: CLI Chat Interface

> [!WARNING]
> We recommend enabling prefix caching, otherwise probing will be very slow.

1. Setup a vLLM server
```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -tp 1 --enable-prefix-caching
```


2. Open Dynasor Chat in command line

```bash
dynasor-chat
```



### `dynasor-openai` OpenAI Compatible Server


1. Setup a vLLM server
```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -tp 1 --enable-prefix-caching
```

2. Setup OpenAI compatible proxy server to server Dynasor
```bash
dynasor-openai
```

3. Use our simiple client script to query:
```bash
# Sample Dynasor client script to ask some questions
python examples/client.py --prompt "2+2=?"
python examples/client.py --prompt "Solve x^2 + 4x = 4"
python examples/client.py --prompt "How many nonzero points are there on x^3y + y^3z + z^3x = 0 over the finite field  𝔽_{{5}^{18}}  up to scaling?"
```


### `dynasor-vllm`: vLLM-native Server

We build Dynasor on top of vLLM as a part of the vLLM OpenAI compatible server endpoint.

1. Setup a dynasor-vllm server
```bash
dynasor-vllm --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --enable-prefix-caching
```

2. Use our simple client script to query:
```bash
python examples/client-vllm.py
```


## Benchmark

### Token Deprivation and Applying Dynasor

To conduct the token deprivation experiment on the math500 dataset, first launch a vLLM server, then run the following command. Note that the current `run.py` script processes only 10 questions. To obtain complete results, modify the --start and --end parameters for changing problem id and solve all problems in parallel!

```bash
bash benchmark/TokenDeprivation/run.sh
```

### Results Visualization

Run `benchmark/TokenDeprivation/post_process.ipynb` to visualize the results

## Citation

If you use Dynasor for your research, please cite our [paper](https://arxiv.org/abs/2412.20993):

```bibtex
@article{fu2024efficiently,
  title={Efficiently Serving LLM Reasoning Programs with Certaindex},
  author={Fu, Yichao and Chen, Junda and Zhu, Siqi and Fu, Zheyu and Dai, Zhongdongming and Qiao, Aurick and Zhang, Hao},
  journal={arXiv preprint arXiv:2412.20993},
  year={2024}
}
```
