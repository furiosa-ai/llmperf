# TODO: Generalize dataset loading method using abstraction class like LLMClient
from typing import Tuple
from llmperf.datasets.gpqa import randomly_sample_gpqa_prompt
from llmperf.datasets.human_eval import randomly_sample_human_eval_prompt
from llmperf.datasets.sonnet import randomly_sample_sonnet_lines_prompt
from llmperf.datasets.translation import randomly_sample_translation_prompt


def randomly_sample_prompt(
    dataset, prompt_tokens_mean, prompt_tokens_stddev, expect_output_tokens, tokenizer
) -> Tuple[str, int]:
    if dataset == "sonnet":
        return randomly_sample_sonnet_lines_prompt(
            prompt_tokens_mean, prompt_tokens_stddev, expect_output_tokens, tokenizer
        )
    elif dataset == "human-eval":
        return randomly_sample_human_eval_prompt(
            prompt_tokens_mean, prompt_tokens_stddev, expect_output_tokens, tokenizer
        )
    elif dataset == "gpqa":
        return randomly_sample_gpqa_prompt(
            prompt_tokens_mean, prompt_tokens_stddev, expect_output_tokens, tokenizer
        )
    elif dataset == "translation":
        return randomly_sample_translation_prompt(
            prompt_tokens_mean, prompt_tokens_stddev, expect_output_tokens, tokenizer
        )
    else:
        raise ValueError(f"Not supported dataset {dataset}")
