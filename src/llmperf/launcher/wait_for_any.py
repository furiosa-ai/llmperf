import time
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
from llmperf.launcher import RequestsLauncher
from llmperf.models import RequestConfig


class WaitForAnyLauncher(RequestsLauncher):
    """RequestsLauncher that waits for any sended request before send a next request.

    The WaitForAny launcher sends a next request when any of requests sent are completed.
    It keeps the number of concurrently processing requests at n.
    """

    def launch(
        self,
        timeout: int,
        prompts: List[Tuple[str, int]],
        num_output_tokens_list: List[int],
    ) -> Tuple[List[Tuple[Dict[str, Any], str, RequestConfig]], int]:
        completed_requests = []
        pbar = tqdm(total=len(prompts))
        start_time = time.monotonic()

        for (prompt, num_output_tokens) in zip(prompts, num_output_tokens_list):
            default_sampling_params = {"max_tokens": num_output_tokens}
            default_sampling_params.update(self._additional_sampling_params)
            request_config = RequestConfig(
                model=self._model,
                prompt=prompt,
                sampling_params=default_sampling_params,
            )
            # Do not care about is there any idle actor.
            # If there is no idle actor, it will be added at pending request
            self._llm_client_pool.submit(
                lambda client, _request_config: client.llm_request.remote(
                    _request_config
                ),
                request_config,
            )

        while (
            time.monotonic() - start_time < timeout and self._llm_client_pool.has_next()
        ):
            completed = self._llm_client_pool.get_next_unordered()
            completed_requests.append(completed)
            pbar.update(1)

        pbar.close()
        end_time = time.monotonic()
        return completed_requests, end_time - start_time
