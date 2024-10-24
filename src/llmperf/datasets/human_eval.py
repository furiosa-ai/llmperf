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
    return [prompt, get_token_length(prompt)]
