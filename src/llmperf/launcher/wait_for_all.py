import time
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
from llmperf.launcher import RequestsLauncher
from llmperf.models import RequestConfig


class WaitForAllLauncher(RequestsLauncher):

    def launch(
        self,
        timeout: int,
        prompts: List[Tuple[str, int]],
        num_output_tokens_list: List[int],
    ) -> Tuple[List[Tuple[Dict[str, Any], str, RequestConfig]], int]:
        completed_requests = []
        max_num_completed_requests = len(prompts)
        pbar = tqdm(total=max_num_completed_requests)
        start_time = time.monotonic()

        while (
            time.monotonic() - start_time < timeout
            and len(completed_requests) < max_num_completed_requests
        ):
            request_configs = []
            for _ in range(
                min(
                    self._num_concurrency_requests,
                    max_num_completed_requests - len(completed_requests),
                )
            ):
                default_sampling_params = {"max_tokens": num_output_tokens_list.pop()}
                default_sampling_params.update(self._additional_sampling_params)
                request_config = RequestConfig(
                    model=self._model,
                    prompt=prompts.pop(),
                    sampling_params=default_sampling_params,
                )
                request_configs.append(request_config)
            completed = list(
                self._llm_client_pool.map_unordered(
                    lambda client, _request_config: client.llm_request.remote(
                        _request_config
                    ),
                    request_configs,
                )
            )
            completed_requests.extend(completed)
            pbar.update(len(completed))

        pbar.close()
        end_time = time.monotonic()
        return completed_requests, end_time - start_time
