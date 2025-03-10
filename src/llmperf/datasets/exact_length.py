from typing import Tuple, List

from llmperf.utils import sample_random_positive_int


def randomly_sample_exact_length_prompt(
    get_token_len=None,
    prompt_tokens_mean: int = 550,
    prompt_tokens_stddev: int = 250,
    expect_output_tokens: int = 150,
    tokenizer=None,
) -> Tuple[List[int], int]:
    num_prompt_tokens = sample_random_positive_int(
        prompt_tokens_mean, prompt_tokens_stddev
    )
    tokens = tokenizer.encode("01" * 100, add_special_tokens=False)
    while len(tokens) < num_prompt_tokens:
        tokens = tokens * 2
    tokens = tokens[:num_prompt_tokens]
    return [tokens, len(tokens)]
