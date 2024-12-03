import csv
import random
import re
from typing import Tuple, Callable

import sys

from llmperf.utils import sample_random_positive_int

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


def clean_text(text) -> str:
    text = text.strip()  # Remove leading and trailing spaces
    text = text.replace(" [title]", ". ")  # Replace specific pattern with period
    text = re.sub(r"\[.*?\]", "", text)  # Remove any text inside brackets
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    return text


def randomly_sample_gpqa_prompt(
    prompt_tokens_mean: int = 550,
    prompt_tokens_stddev: int = 250,
    expect_output_tokens: int = 150,
    get_token_len=Callable[[str], int],
) -> Tuple[str, int]:
    prompts = []
    # FIXME: Use gpqa_other dataset
    min_prompt_length = sys.maxint
    gpqa_main = csv.DictReader(open("gpqa_main.csv"))
    for doc in gpqa_main:
        # copy permutation code from AA's sample code
        choices = [
            doc["Correct Answer"],
            doc["Incorrect Answer 1"],
            doc["Incorrect Answer 2"],
            doc["Incorrect Answer 3"],
        ]
        choices = [choices[i] for i in doc["permutation"]]
        choices_dict = dict(
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
            Question=doc["Question"],  # Added preprocess here
        )
        prompt = QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)
        prompts.append(prompt)

        if (token_len := get_token_len(prompt)) < min_prompt_length:
            min_prompt_length = token_len

    num_prompt_tokens = sample_random_positive_int(
        prompt_tokens_mean, prompt_tokens_stddev
    )
    while num_prompt_tokens < min_prompt_length:
        num_prompt_tokens = sample_random_positive_int(
            prompt_tokens_mean, prompt_tokens_stddev
        )

    prompt = random.choice(prompts)
    while num_prompt_tokens < get_token_len(prompt):
        prompt = random.choice(prompts)

    # padding
    # pad_token_num = 0
    # remaining_prompt_tokens = num_prompt_tokens - get_token_len(prompt)
    # while remaining_prompt_tokens > 0:
    #     pad_token_num += 1
    #     remaining_prompt_tokens -= get_token_len(tokenizer.pad_token * pad_token_num)
    # prompt += tokenizer.pad_token * (pad_token_num - 1)

    return [prompt, get_token_len(prompt)]
