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
"""DetokenizerHandler handles only the detokenization logic for paired workers."""

import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Union

import zmq

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOut,
    BatchMultimodalDecodeReq,
    BatchMultimodalOut,
    BatchStrOut,
    BatchTokenIDOut,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_zmq_socket
from sglang.utils import find_printable_text

logger = logging.getLogger(__name__)

# Maximum number of request states that detokenizer can hold
DETOKENIZER_MAX_STATES = int(os.environ.get("SGLANG_DETOKENIZER_MAX_STATES", 1 << 16))


@dataclass
class DecodeStatus:
    """Store the status of incremental decoding."""

    decoded_text: str
    decode_ids: List[int]
    surr_offset: int
    read_offset: int
    # Offset that's sent to tokenizer for incremental update.
    sent_offset: int = 0


class DetokenizerHandler:
    """Handles only the detokenization logic for paired workers."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        worker_id: int = 0,
    ):
        self.worker_id = worker_id
        self.server_args = server_args

        # Init inter-process communication
        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
        )
        self.send_to_tokenizer_manager = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_manager_ipc_name, False
        )

        # Initialize tokenizer for detokenization
        if server_args.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )

        # Decode status tracking
        self.decode_status = LimitedCapacityDict(capacity=DETOKENIZER_MAX_STATES)

        # Performance monitoring
        self.performance_stats = {
            "total_requests": 0,
            "total_tokens_processed": 0,
            "total_processing_time": 0.0,
            "enable_worker_logging": getattr(
                server_args, "enable_detokenizer_worker_logging", False
            ),
        }

        logger.info(
            f"🔌 DetokenizerHandler {self.worker_id} - 🚀 Initialized successfully"
        )

    def run(self):
        """Run the detokenizer handler."""
        start_time = time.time()
        logger.info(f"🔌 DetokenizerHandler {self.worker_id} - 🚀 Starting...")

        try:
            self.event_loop()
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(
                f"🔌 DetokenizerHandler {self.worker_id} - ❌ Failed after {total_time:.2f}s: {e}"
            )
            raise
        finally:
            total_time = time.time() - start_time
            logger.info(
                f"🔌 DetokenizerHandler {self.worker_id} - 🛑 Stopped after {total_time:.2f}s"
            )

    def event_loop(self):
        """The event loop that handles detokenization requests"""
        while True:
            start_time = time.time()

            # Receive request from scheduler
            recv_start = time.time()
            recv_obj = self.recv_from_scheduler.recv_pyobj()
            recv_time = time.time() - recv_start

            # Process the detokenization
            process_start = time.time()
            output = self._process_detokenization(recv_obj)
            process_time = time.time() - process_start

            # Send result to paired TokenizerManager
            send_start = time.time()
            self.send_to_tokenizer_manager.send_pyobj(output)
            send_time = time.time() - send_start

            # Log performance if enabled
            if self.performance_stats["enable_worker_logging"]:
                total_time = time.time() - start_time
                logger.info(
                    f"🔌 DetokenizerHandler {self.worker_id} - ✅ Completed cycle in {total_time:.4f}s:\n"
                    f"   - Receive: {recv_time:.4f}s\n"
                    f"   - Process: {process_time:.4f}s\n"
                    f"   - Send: {send_time:.4f}s"
                )

            # Update performance stats
            self._update_performance_stats(recv_obj, process_time)

    def _process_detokenization(self, recv_obj):
        """Process the detokenization request and return the result."""
        if isinstance(recv_obj, BatchTokenIDOut):
            return self._handle_batch_token_id_out(recv_obj)
        elif isinstance(recv_obj, BatchEmbeddingOut):
            return self._handle_batch_embedding_out(recv_obj)
        elif isinstance(recv_obj, BatchMultimodalDecodeReq):
            return self._handle_multimodal_decode_req(recv_obj)
        else:
            logger.error(
                f"🔌 DetokenizerHandler {self.worker_id} - Unknown request type: {type(recv_obj)}"
            )
            return recv_obj

    def _handle_batch_token_id_out(self, recv_obj: BatchTokenIDOut):
        """Handle BatchTokenIDOut detokenization."""
        start_time = time.time()
        bs = len(recv_obj.rids)

        if self.performance_stats["enable_worker_logging"]:
            logger.info(
                f"🔌 DetokenizerHandler {self.worker_id} - 🔤 Processing BatchTokenIDOut: {bs} requests"
            )

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

        # Perform detokenization
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
                    f"Decode status not found for request {recv_obj.rids[i]}"
                )

            new_text = read_texts[i][len(surr_texts[i]) :]
            if recv_obj.finished_reasons[i] is None:
                # Streaming chunk: update the decode status
                if len(new_text) > 0 and not new_text.endswith(""):
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

            # Incrementally send text
            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)
            output_strs.append(incremental_output)

        # Clean up finished requests
        for i, rid in enumerate(recv_obj.rids):
            if recv_obj.finished_reasons[i] is not None:
                if rid in self.decode_status:
                    del self.decode_status[rid]

        total_time = time.time() - start_time
        if self.performance_stats["enable_worker_logging"]:
            logger.info(
                f"🔌 DetokenizerHandler {self.worker_id} - ✅ BatchTokenIDOut completed in {total_time:.4f}s"
            )

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

    def _handle_batch_embedding_out(self, recv_obj: BatchEmbeddingOut):
        """Handle BatchEmbeddingOut - no detokenization needed."""
        if self.performance_stats["enable_worker_logging"]:
            logger.info(
                f"🔌 DetokenizerHandler {self.worker_id} - 📊 Processing BatchEmbeddingOut: {len(recv_obj.rids)} requests"
            )
        return recv_obj

    def _handle_multimodal_decode_req(self, recv_obj: BatchMultimodalDecodeReq):
        """Handle multimodal decode requests."""
        if self.performance_stats["enable_worker_logging"]:
            logger.info(
                f"🔌 DetokenizerHandler {self.worker_id} - 🖼️ Processing BatchMultimodalDecodeReq: {len(recv_obj.rids)} requests"
            )

        outputs = self.tokenizer.detokenize(recv_obj)

        return BatchMultimodalOut(
            rids=recv_obj.rids,
            finished_reasons=recv_obj.finished_reasons,
            outputs=outputs,
            prompt_tokens=recv_obj.prompt_tokens,
            completion_tokens=recv_obj.completion_tokens,
            cached_tokens=recv_obj.cached_tokens,
        )

    def trim_matched_stop(
        self, output: Union[str, List[int]], finished_reason: Dict, no_stop_trim: bool
    ):
        """Trim matched stop tokens/strings."""
        if no_stop_trim or not finished_reason:
            return output

        matched = finished_reason.get("matched", None)
        if not matched:
            return output

        # Trim stop string
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)
            return output[:pos] if pos != -1 else output

        # Trim stop token
        if isinstance(matched, int) and isinstance(output, list):
            assert len(output) > 0
            return output[:-1]

        return output

    def _update_performance_stats(self, recv_obj, processing_time: float):
        """Update performance statistics."""
        self.performance_stats["total_requests"] += 1
        self.performance_stats["total_processing_time"] += processing_time

        # Count tokens if available
        if hasattr(recv_obj, "decode_ids"):
            total_tokens = sum(len(decode_ids) for decode_ids in recv_obj.decode_ids)
            self.performance_stats["total_tokens_processed"] += total_tokens


class LimitedCapacityDict(OrderedDict):
    """Dictionary with limited capacity that evicts oldest items."""

    def __init__(self, capacity: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity

    def __setitem__(self, key, value):
        if len(self) >= self.capacity:
            # Remove the oldest element
            self.popitem(last=False)
        super().__setitem__(key, value)


def run_detokenizer_handler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    worker_id: int = 0,
):
    """Run DetokenizerHandler in a separate process."""
    import psutil
    import setproctitle

    setproctitle.setproctitle(f"sglang::detokenizer_handler_{worker_id}")

    try:
        handler = DetokenizerHandler(server_args, port_args, worker_id)
        handler.run()
    except Exception as e:
        logger.error(f"🔌 DetokenizerHandler {worker_id} failed: {e}")
        raise
