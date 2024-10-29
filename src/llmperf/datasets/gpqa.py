import csv
import random
import re
from typing import Tuple
from transformers import AutoTokenizer

import sys

from llmperf.utils import sample_random_positive_int

QUERY_TEMPLATE = """What is the correct answer to this question:
{Question}

Choices:
(A) {A}
(B) {B}
(C) {C}
(D) {D}

Answer: """


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
    tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct"),
) -> Tuple[str, int]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    get_token_length = lambda text: len(tokenizer.encode(text))

    prompts = []
    # FIXME: Use gpqa_other dataset
    min_prompt_length = sys.maxint
    gpqa_main = csv.DictReader(open("gpqa_main.csv"))
    for doc in gpqa_main:
        choices = [
            clean_text(doc["Correct Answer"]),
            clean_text(doc["Incorrect Answer 1"]),
            clean_text(doc["Incorrect Answer 2"]),
            clean_text(doc["Incorrect Answer 3"]),
        ]
        random.shuffle(choices)
        choices_dict = dict(
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
            Question=clean_text(doc["Question"]),  # Added preprocess here
        )
        prompt = QUERY_TEMPLATE.format(**choices_dict)
        prompts.append(prompt)

        if get_token_length(prompt) < min_prompt_length:
            min_prompt_length = get_token_length(prompt)

    num_prompt_tokens = sample_random_positive_int(
        prompt_tokens_mean, prompt_tokens_stddev
    )
    while num_prompt_tokens < min_prompt_length:
        num_prompt_tokens = sample_random_positive_int(
            prompt_tokens_mean, prompt_tokens_stddev
        )

    prompt = random.choice(prompts)
    while num_prompt_tokens < get_token_length(prompt):
        prompt = random.choice(prompts)
    remaining_prompt_tokens = num_prompt_tokens - get_token_length(prompt)

    # padding
    pad_token_num = 0
    while remaining_prompt_tokens > 0:
        pad_token_num += 1
        remaining_prompt_tokens -= get_token_length(tokenizer.pad_token * pad_token_num)
    prompt += tokenizer.pad_token * (pad_token_num - 1)

    return [prompt, num_prompt_tokens]
