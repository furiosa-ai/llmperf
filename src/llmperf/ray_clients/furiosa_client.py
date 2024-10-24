import json
import time
from typing import Any, Dict

import requests
from llmperf import common_metrics
from llmperf.models import RequestConfig
from llmperf.ray_llm_client import LLMClient
import ray


@ray.remote
class FuriosaLLMClient(LLMClient):
    """Client for FuriosaAI LLM Completion API."""

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        prompt = request_config.prompt
        prompt, prompt_len = prompt

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        generated_text = ""
        output_throughput = 0
        total_request_time = 0

        metrics = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        # FIXME: Change url
        url = "http://localhost:8000" + "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        # TODO: Use model name from request_config
        data = {"model": "EMPTY", "messages": [{"role": "user", "content": prompt}]}

        start_time = time.monotonic()
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            total_request_time = time.monotonic() - start_time
            ttft = total_request_time
            response_code = response.status_code

            response = response.json()
            for choice in response["choices"]:
                generated_text += choice["message"]["content"]
            tokens_received = response["usage"]["completion_tokens"]
            output_throughput = tokens_received / total_request_time
            time_to_next_token = [
                total_request_time / tokens_received for _ in range(tokens_received)
            ]

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = str(e)
            metrics[common_metrics.ERROR_CODE] = response_code
            print(f"Warning Or Error: {e}")
            print(response_code)
            print(response_code)

        metrics[common_metrics.INTER_TOKEN_LAT] = sum(
            time_to_next_token
        )  # This should be same as metrics[common_metrics.E2E_LAT]. Leave it here for now
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config
