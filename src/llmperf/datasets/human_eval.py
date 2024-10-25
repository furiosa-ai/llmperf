import random
from typing import Tuple
from transformers import AutoTokenizer
from human_eval.data import read_problems

from llmperf.utils import sample_random_positive_int


def randomly_sample_human_eval_prompt(
    prompt_tokens_mean: int = 550,
    prompt_tokens_stddev: int = 250,
    expect_output_tokens: int = 150,
    tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"),
) -> Tuple[str, int]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    get_token_length = lambda text: len(tokenizer.encode(text))

    # Leave the prompt empty for now
    prompt = ""

    # FIXME: How should we control input/output prompt length about human_eval dataset?
    num_prompt_tokens = sample_random_positive_int(
        prompt_tokens_mean, prompt_tokens_stddev
    )
    while num_prompt_tokens < get_token_length(prompt):
        num_prompt_tokens = sample_random_positive_int(
            prompt_tokens_mean, prompt_tokens_stddev
        )
    remaining_prompt_tokens = num_prompt_tokens - get_token_length(prompt)
    problems = read_problems()
    task_ids = list(problems.keys())
    task_id = random.choice(task_ids)
    while remaining_prompt_tokens < get_token_length(problems[task_id]["prompt"]):
        task_id = random.choice(task_ids)
    prompt += problems[task_id]["prompt"]
    remaining_prompt_tokens -= get_token_length(prompt)

    # padding
    pad_token_num = 0
    while remaining_prompt_tokens > 0:
        pad_token_num += 1
        remaining_prompt_tokens -= get_token_length(tokenizer.pad_token * pad_token_num)
    prompt += tokenizer.pad_token * (pad_token_num - 1)

    return [prompt, num_prompt_tokens]
