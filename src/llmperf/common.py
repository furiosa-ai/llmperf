from typing import Any, Dict, List, Callable
from llmperf.launcher import RequestsLauncher
from llmperf.launcher.wait_for_all import WaitForAllLauncher
from llmperf.launcher.wait_for_any import WaitForAnyLauncher
from llmperf.ray_clients.furiosa_client import FuriosaLLMClient
from llmperf.ray_clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from llmperf.ray_clients import LLMClient


SUPPORTED_APIS = ["openai", "furiosa"]


def construct_clients(
    llm_api: str, num_clients: int, get_token_len: Callable[[str], int]
) -> List[LLMClient]:
    """Construct LLMClients that will be used to make requests to the LLM API.

    Args:
        llm_api: The name of the LLM API to use.
        num_clients: The number of concurrent requests to make.

    Returns:
        The constructed LLMCLients

    """
    if llm_api == "openai":
        clients = [
            OpenAIChatCompletionsClient.remote(get_token_len)
            for _ in range(num_clients)
        ]
    elif llm_api == "furiosa":
        clients = [FuriosaLLMClient.remote(get_token_len) for _ in range(num_clients)]
    else:
        raise ValueError(
            f"llm_api must be one of the supported LLM APIs: {SUPPORTED_APIS}"
        )

    return clients


def construct_launcher(
    wait_for: str,
    model: str,
    clients: List[LLMClient],
    additional_sampling_params: Dict[str, Any],
) -> RequestsLauncher:
    """Construct RequestsLauncher that will send requests with a specific pattern.

    Args:
        wait_for: The name of pattern. WaitForAll launcher
        model: The name of the model to query.
        clients: The list of LLMClients.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions

    Returns:
        The constructed RequesstLauncher

    """
    if wait_for == "all":
        return WaitForAllLauncher(model, clients, additional_sampling_params)
    elif wait_for == "any":
        return WaitForAnyLauncher(model, clients, additional_sampling_params)
    else:
        raise ValueError(f"Wrong type for 'wait_for' option: {wait_for}")
