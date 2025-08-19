# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DetokenizerManager is a process that detokenizes the token ids."""

import dataclasses
import logging
import os
import signal
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import psutil
import setproctitle
import zmq

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.base_manager import BaseManager
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOut,
    BatchMultimodalDecodeReq,
    BatchMultimodalOut,
    BatchStrOut,
    BatchTokenIDOut,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.srt_py_object import SrtPyObject
from sglang.srt.utils import (
    configure_logger,
    get_scheduler_manager,
    get_zmq_socket,
    kill_itself_when_parent_died,
)
from sglang.utils import (
    TypeBasedDispatcher,
    find_printable_text,
    get_exception_traceback,
)

logger = logging.getLogger(__name__)

# Maximum number of request states that detokenizer can hold. When exceeded,
# oldest request states will be evicted. Default: 65536 (1<<16).
# For more details, see: https://github.com/sgl-project/sglang/issues/2812
# Use power of 2 values for better memory allocation.
DETOKENIZER_MAX_STATES = int(os.environ.get("SGLANG_DETOKENIZER_MAX_STATES", 1 << 16))


@dataclasses.dataclass
class DecodeStatus:
    """Store the status of incremental decoding."""

    decoded_text: str
    decode_ids: List[int]
    surr_offset: int
    read_offset: int
    # Offset that's sent to tokenizer for incremental update.
    sent_offset: int = 0


class DetokenizerManager(BaseManager):
    """Manager for detokenization process."""

    def __init__(
        self,
        server_args,
        port_args,
        worker_id: int = 0,
    ):
        super().__init__(server_args, port_args)
        self.worker_id = worker_id
        self.scheduler_manager = None
        logger.info(f"🔧 DetokenizerManager {worker_id} initialized")

    def run(self):
        """Run the detokenizer manager."""
        start_time = time.time()
        logger.info(f"🚀 DetokenizerManager {self.worker_id} starting...")

        # Time the scheduler connection
        scheduler_start = time.time()
        self.scheduler_manager = get_scheduler_manager(self.port_args)
        scheduler_time = time.time() - scheduler_start
        logger.info(
            f"🔌 DetokenizerManager {self.worker_id} connected to scheduler in {scheduler_time:.4f}s"
        )

        # Time the main processing loop
        loop_start = time.time()
        try:
            while True:
                loop_iteration_start = time.time()

                # Time the receive operation
                receive_start = time.time()
                data = self.scheduler_manager.recv_pyobj()
                receive_time = time.time() - receive_start

                if data is None:
                    logger.info(
                        f"🔍 DetokenizerManager {self.worker_id} received None, continuing..."
                    )
                    continue

                # Time the processing operation
                process_start = time.time()
                self._process_data(data)
                process_time = time.time() - process_start

                loop_iteration_time = time.time() - loop_iteration_start

                logger.debug(
                    f"🔄 DetokenizerManager {self.worker_id} iteration completed in {loop_iteration_time:.4f}s:\n"
                    f"   - Receive: {receive_time:.4f}s\n"
                    f"   - Process: {process_time:.4f}s\n"
                    f"   - Data type: {type(data).__name__}"
                )

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(
                f"❌ DetokenizerManager {self.worker_id} failed after {total_time:.2f}s: {e}\n"
                f"   - Exception type: {type(e).__name__}"
            )
            raise
        finally:
            total_time = time.time() - start_time
            logger.info(
                f"🛑 DetokenizerManager {self.worker_id} stopped after {total_time:.2f}s"
            )

    def _process_data(self, data: SrtPyObject):
        """Process received data."""
        start_time = time.time()
        logger.info(
            f"🔍 DetokenizerManager {self.worker_id} processing data: {type(data).__name__}"
        )

        # Time the data type check
        type_check_start = time.time()
        if hasattr(data, "output_ids"):
            # Token generation output
            logger.debug(
                f"🔤 Processing token generation output for {len(data.rids)} requests"
            )
            output = BatchTokenIDOut(
                rids=data.rids,
                output_ids=data.output_ids,
                logprobs=data.logprobs,
                finish_reasons=data.finish_reasons,
            )
        elif hasattr(data, "embeddings"):
            # Embedding output
            logger.debug(
                f"📊 Processing embedding output for {len(data.rids)} requests"
            )
            output = BatchEmbeddingOut(
                rids=data.rids,
                embeddings=data.embeddings,
            )
        else:
            logger.warning(f"⚠️ Unknown data type: {type(data)}")
            return

        type_check_time = time.time() - type_check_start

        # Time the response sending
        send_start = time.time()
        try:
            self.scheduler_manager.send_pyobj(output)
            send_time = time.time() - send_start
            total_time = time.time() - start_time

            logger.info(
                f"✅ DetokenizerManager {self.worker_id} completed processing in {total_time:.4f}s:\n"
                f"   - Type check: {type_check_time:.4f}s\n"
                f"   - Send response: {send_time:.4f}s\n"
                f"   - Requests: {len(data.rids)}, RIDs: {data.rids[:3]}{'...' if len(data.rids) > 3 else ''}"
            )

        except Exception as e:
            send_time = time.time() - send_start
            total_time = time.time() - start_time

            logger.error(
                f"❌ DetokenizerManager {self.worker_id} failed to send response: {e}\n"
                f"   - Total time: {total_time:.4f}s\n"
                f"   - Type check: {type_check_time:.4f}s\n"
                f"   - Send attempt: {send_time:.4f}s\n"
                f"   - Exception type: {type(e).__name__}"
            )
            raise

    def _count_tokens_in_request(self, recv_obj) -> int:
        """Count the number of tokens in a request for performance monitoring"""
        try:
            if hasattr(recv_obj, "decode_ids"):
                # BatchTokenIDOut
                total_tokens = 0
                for decode_ids in recv_obj.decode_ids:
                    total_tokens += len(decode_ids)
                return total_tokens
            elif hasattr(recv_obj, "output_ids"):
                # BatchEmbeddingOut or other types
                return len(recv_obj.output_ids) if recv_obj.output_ids else 0
            else:
                return 1  # Default to 1 if we can't determine
        except Exception:
            return 1  # Fallback to 1 on error

    def trim_matched_stop(
        self, output: Union[str, List[int]], finished_reason: Dict, no_stop_trim: bool
    ):
        if no_stop_trim or not finished_reason:
            return output

        matched = finished_reason.get("matched", None)
        if not matched:
            return output

        # TODO(lmzheng): handle the case where multiple stop strs are hit

        # Trim stop str.
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)
            return output[:pos] if pos != -1 else output

        # Trim stop token.
        if isinstance(matched, int) and isinstance(output, list):
            assert len(output) > 0
            return output[:-1]
        return output

    def handle_batch_embedding_out(self, recv_obj: BatchEmbeddingOut):
        # If it is embedding model, no detokenization is needed.
        return recv_obj

    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOut):
        bs = len(recv_obj.rids)

        # Initialize decode status
        read_ids, surr_ids = [], []
        for i in range(bs):
            rid = recv_obj.rids[i]
            if rid not in self.decode_status:
                s = DecodeStatus(
                    decoded_text=recv_obj.decoded_texts[i],
                    decode_ids=recv_obj.decode_ids[i],
                    surr_offset=0,
                    read_offset=recv_obj.read_offsets[i],
                )
                self.decode_status[rid] = s
            else:
                s = self.decode_status[rid]
                s.decode_ids.extend(recv_obj.decode_ids[i])

            read_ids.append(
                self.trim_matched_stop(
                    s.decode_ids[s.surr_offset :],
                    recv_obj.finished_reasons[i],
                    recv_obj.no_stop_trim[i],
                )
            )
            surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])

        # TODO(lmzheng): handle skip_special_tokens/spaces_between_special_tokens per request
        surr_texts = self.tokenizer.batch_decode(
            surr_ids,
            skip_special_tokens=recv_obj.skip_special_tokens[0],
            spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[0],
        )
        read_texts = self.tokenizer.batch_decode(
            read_ids,
            skip_special_tokens=recv_obj.skip_special_tokens[0],
            spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[0],
        )

        # Incremental decoding
        output_strs = []
        for i in range(bs):
            try:
                s = self.decode_status[recv_obj.rids[i]]
            except KeyError:
                raise RuntimeError(
                    f"Decode status not found for request {recv_obj.rids[i]}. "
                    "It may be due to the request being evicted from the decode status due to memory pressure. "
                    "Please increase the maximum number of requests by setting "
                    "the SGLANG_DETOKENIZER_MAX_STATES environment variable to a bigger value than the default value. "
                    f"The current value is {DETOKENIZER_MAX_STATES}. "
                    "For more details, see: https://github.com/sgl-project/sglang/issues/2812"
                )
            new_text = read_texts[i][len(surr_texts[i]) :]
            if recv_obj.finished_reasons[i] is None:
                # Streaming chunk: update the decode status
                if len(new_text) > 0 and not new_text.endswith("�"):
                    s.decoded_text = s.decoded_text + new_text
                    s.surr_offset = s.read_offset
                    s.read_offset = len(s.decode_ids)
                    new_text = ""
                else:
                    new_text = find_printable_text(new_text)

            output_str = self.trim_matched_stop(
                s.decoded_text + new_text,
                recv_obj.finished_reasons[i],
                recv_obj.no_stop_trim[i],
            )
            # Incrementally send text.
            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)
            output_strs.append(incremental_output)

        return BatchStrOut(
            rids=recv_obj.rids,
            finished_reasons=recv_obj.finished_reasons,
            output_strs=output_strs,
            output_ids=recv_obj.output_ids,
            prompt_tokens=recv_obj.prompt_tokens,
            completion_tokens=recv_obj.completion_tokens,
            cached_tokens=recv_obj.cached_tokens,
            spec_verify_ct=recv_obj.spec_verify_ct,
            input_token_logprobs_val=recv_obj.input_token_logprobs_val,
            input_token_logprobs_idx=recv_obj.input_token_logprobs_idx,
            output_token_logprobs_val=recv_obj.output_token_logprobs_val,
            output_token_logprobs_idx=recv_obj.output_token_logprobs_idx,
            input_top_logprobs_val=recv_obj.input_top_logprobs_val,
            input_top_logprobs_idx=recv_obj.input_top_logprobs_idx,
            output_top_logprobs_val=recv_obj.output_top_logprobs_val,
            output_top_logprobs_idx=recv_obj.output_top_logprobs_idx,
            input_token_ids_logprobs_val=recv_obj.input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=recv_obj.input_token_ids_logprobs_idx,
            output_token_ids_logprobs_val=recv_obj.output_token_ids_logprobs_val,
            output_token_ids_logprobs_idx=recv_obj.output_token_ids_logprobs_idx,
            output_hidden_states=recv_obj.output_hidden_states,
        )

    def handle_multimodal_decode_req(self, recv_obj: BatchMultimodalDecodeReq):
        outputs = self.tokenizer.detokenize(recv_obj)
        return BatchMultimodalOut(
            rids=recv_obj.rids,
            finished_reasons=recv_obj.finished_reasons,
            outputs=outputs,
            prompt_tokens=recv_obj.prompt_tokens,
            completion_tokens=recv_obj.completion_tokens,
            cached_tokens=recv_obj.cached_tokens,
        )


class LimitedCapacityDict(OrderedDict):
    def __init__(self, capacity: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity

    def __setitem__(self, key, value):
        if len(self) >= self.capacity:
            # Remove the oldest element (first item in the dict)
            self.popitem(last=False)
        # Set the new item
        super().__setitem__(key, value)


def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    worker_id: int = 0,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle(f"sglang::detokenizer_worker_{worker_id}")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        manager = DetokenizerManager(server_args, port_args, worker_id)
        manager.run()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(
            f"🔌 Worker {worker_id} - DetokenizerManager hit an exception: {traceback}"
        )
        parent_process.send_signal(signal.SIGQUIT)
