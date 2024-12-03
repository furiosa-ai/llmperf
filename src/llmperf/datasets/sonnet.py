import math
import pathlib
import random
from typing import Tuple, Callable

from llmperf.utils import sample_random_positive_int


def randomly_sample_sonnet_lines_prompt(
    prompt_tokens_mean: int = 550,
    prompt_tokens_stddev: int = 250,
    expect_output_tokens: int = 150,
    get_token_len=Callable[[str], int],
) -> Tuple[str, int]:
    """Generate a prompt that randomly samples lines from a the shakespeare sonnet at sonnet.txt.

    Args:
        prompt_length_mean: The mean length of the prompt to generate.
        prompt_len_stddev: The standard deviation of the length of the prompt to generate.
        expect_output_tokens: The number of tokens to expect in the output. This is used to
        determine the length of the prompt. The prompt will be generated such that the output
        will be approximately this many tokens.

    Note:
        tokens will be counted from the sonnet using the Llama tokenizer. Using one tokenizer
        ensures a fairer comparison across different LLMs. For example, if gpt 3.5 tokenizes
        a prompt in less tokens than Llama2, then this will be reflected in the results since
        they will be fed identical prompts.

    Returns:
        A tuple of the prompt and the length of the prompt.
    """

    prompt = (
        "Randomly stream lines from the following text "
        f"with {expect_output_tokens} output tokens. "
        "Don't generate eos tokens:\n\n"
    )
    # get a prompt length that is at least as long as the base
    num_prompt_tokens = sample_random_positive_int(
        prompt_tokens_mean, prompt_tokens_stddev
    )
    while num_prompt_tokens < get_token_len(prompt):
        num_prompt_tokens = sample_random_positive_int(
            prompt_tokens_mean, prompt_tokens_stddev
        )
    remaining_prompt_tokens = num_prompt_tokens - get_token_len(prompt)
    sonnet_path = pathlib.Path(__file__).parent.resolve() / "sonnet.txt"
    with open(sonnet_path, "r") as f:
        sonnet_lines = f.readlines()
    random.shuffle(sonnet_lines)
    sampling_lines = True
    while sampling_lines:
        for line in sonnet_lines:
            line_to_add = line
            if remaining_prompt_tokens - get_token_len(line_to_add) < 0:
                # This will cut off a line in the middle of a word, but that's ok since an
                # llm should be able to handle that.
                line_to_add = line_to_add[: int(math.ceil(remaining_prompt_tokens))]
                sampling_lines = False
                prompt += line_to_add
                break
            prompt += line_to_add
            remaining_prompt_tokens -= get_token_len(line_to_add)
    return (prompt, num_prompt_tokens)
