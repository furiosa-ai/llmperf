# TODO: Generalize dataset loading method using abstraction class like LLMClient
from typing import Tuple
from llmperf.datasets.gpqa import randomly_sample_gpqa_prompt
from llmperf.datasets.sonnet import randomly_sample_sonnet_lines_prompt
from llmperf.datasets.translation import randomly_sample_translation_prompt


def randomly_sample_prompt(
    dataset,
    prompt_tokens_mean,
    prompt_tokens_stddev,
    expect_output_tokens,
    get_token_len,
) -> Tuple[str, int]:
    if dataset == "sonnet":
        f = randomly_sample_sonnet_lines_prompt
    elif dataset == "human-eval":
        from llmperf.datasets.human_eval import randomly_sample_human_eval_prompt

        f = randomly_sample_human_eval_prompt
    elif dataset == "gpqa":
        f = randomly_sample_gpqa_prompt
    elif dataset == "translation":
        f = randomly_sample_translation_prompt
    else:
        raise ValueError(f"Not supported dataset {dataset}")

    return f(
        get_token_len,
        prompt_tokens_mean,
        prompt_tokens_stddev,
        expect_output_tokens,
    )
