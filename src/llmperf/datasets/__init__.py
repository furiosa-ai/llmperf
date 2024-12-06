# TODO: Generalize dataset loading method using abstraction class like LLMClient
from typing import Tuple


def randomly_sample_prompt(
    dataset,
    prompt_tokens_mean,
    prompt_tokens_stddev,
    expect_output_tokens,
    get_token_len,
) -> Tuple[str, int]:
    if dataset == "sonnet":
        from llmperf.datasets.sonnet import randomly_sample_sonnet_lines_prompt

        f = randomly_sample_sonnet_lines_prompt
    elif dataset == "human-eval":
        from llmperf.datasets.human_eval import randomly_sample_human_eval_prompt

        f = randomly_sample_human_eval_prompt
    elif dataset == "gpqa":
        from llmperf.datasets.gpqa import randomly_sample_gpqa_prompt

        f = randomly_sample_gpqa_prompt
    elif dataset == "translation":
        from llmperf.datasets.translation import randomly_sample_translation_prompt

        f = randomly_sample_translation_prompt
    else:
        raise ValueError(f"Not supported dataset {dataset}")

    return f(
        get_token_len,
        prompt_tokens_mean,
        prompt_tokens_stddev,
        expect_output_tokens,
    )
