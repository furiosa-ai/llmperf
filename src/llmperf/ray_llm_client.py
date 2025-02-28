import abc
from typing import Any, Dict, Tuple

from llmperf.models import RequestConfig


class LLMClient:
    """A client for making requests to a LLM API e.g Anyscale Endpoints."""

    get_token_len = None

    @abc.abstractmethod
    def llm_request(
        self, request_config: RequestConfig
    ) -> Tuple[Dict[str, Any], str, RequestConfig]:
        """Make a single completion request to a LLM API

        Returns:
            Metrics about the performance charateristics of the request.
            The text generated by the request to the LLM API.
            The request_config used to make the request. This is mainly for logging purposes.

        """
        ...
