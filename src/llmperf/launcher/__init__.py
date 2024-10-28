import abc
from typing import Any, List, Dict, Tuple

from llmperf.models import RequestConfig
from llmperf.ray_llm_client import LLMClient
from ray.util import ActorPool

SUPPROTED_SCENARIO = ["multi-stream", "single-stream"]


class RequestsLauncher:
    """Launch requests from LLMClients to their respective LLM APIs."""

    def __init__(
        self,
        model: str,
        llm_clients: List[LLMClient],
        additional_sampling_params: Dict[str, Any],
    ):
        self._model = model
        self._llm_client_pool = ActorPool(llm_clients)
        self._additional_sampling_params = additional_sampling_params
        self._num_concurrency_requests = len(llm_clients)

    @abc.abstractmethod
    def launch(
        self,
        timeout: int,
        prompts: List[Tuple[str, int]],
        num_output_tokens_list: List[int],
    ) -> Tuple[List[Tuple[Dict[str, Any], str, RequestConfig]], int]:
        """Depending on the scenario, send a request to the llm client and receive completion."""
