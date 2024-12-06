import os
from typing import Any, Dict

from llmperf.models import RequestConfig
from llmperf.ray_clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from llmperf.ray_clients import LLMClient
import ray
from ray.runtime_env import RuntimeEnv


FURIOSA_AI_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

@ray.remote
class FuriosaLLMClient(LLMClient):
    """Client for FuriosaAI LLM Completion API."""

    def __init__(self, get_token_len):
        self.get_token_len = get_token_len

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        """
        FuriosaAI LLM Completion API is compatible with OpenAIChatCompletion API.
        So, set Furiosa API base and key as OpenAI API base and key to use OpenAIChatCompletionsClient.
        """

        address = os.environ.get("FURIOSA_API_BASE")
        if not address:
            raise ValueError("the environment variable FURIOSA_API_BASE must be set.")
        key = os.environ.get("FURIOSA_API_KEY")
        if not key:
            # FIXME: Raise value error after api key validation is added
            key = "0000"
        os.environ["OPENAI_API_BASE"] = address
        os.environ["OPENAI_API_KEY"] = key

        # Use greedy search as default
        if "temperature" not in request_config.sampling_params:
            request_config.sampling_params["temperature"] = 0.0
        request_config.system_prompt = FURIOSA_AI_DEFAULT_SYSTEM_PROMPT 

        actor = OpenAIChatCompletionsClient.options(
            runtime_env=RuntimeEnv(env_vars=dict(os.environ))
        ).remote(self.get_token_len)
        future = actor.llm_request.remote(request_config)
        return ray.get(future)
