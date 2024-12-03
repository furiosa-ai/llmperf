import random
from typing import Tuple, Callable
from human_eval.data import read_problems

from llmperf.utils import sample_random_positive_int


def randomly_sample_human_eval_prompt(
    get_token_len: Callable[[str], int],
    prompt_tokens_mean: int = 550,
    prompt_tokens_stddev: int = 250,
    expect_output_tokens: int = 150,
) -> Tuple[str, int]:
    # Instruction from AA's sample code
    prompt = "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n"

    # FIXME: How should we control input/output prompt length about human_eval dataset?
    num_prompt_tokens = sample_random_positive_int(
        prompt_tokens_mean, prompt_tokens_stddev
    )
    while num_prompt_tokens < get_token_len(prompt):
        num_prompt_tokens = sample_random_positive_int(
            prompt_tokens_mean, prompt_tokens_stddev
        )
    remaining_prompt_tokens = num_prompt_tokens - get_token_len(prompt)
    problems = read_problems()
    task_ids = list(problems.keys())
    task_id = random.choice(task_ids)
    while remaining_prompt_tokens < get_token_len(problems[task_id]["prompt"]):
        task_id = random.choice(task_ids)
    prompt += problems[task_id]["prompt"]

    # padding
    # remaining_prompt_tokens -= get_token_length(prompt)
    # pad_token_num = 0
    # while remaining_prompt_tokens > 0:
    #     pad_token_num += 1
    #     remaining_prompt_tokens -= get_token_length(tokenizer.pad_token * pad_token_num)
    # prompt += tokenizer.pad_token * (pad_token_num - 1)

    return [prompt, get_token_len(prompt)]
