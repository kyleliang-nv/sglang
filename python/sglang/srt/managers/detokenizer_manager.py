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
from typing import Dict, List, Union

import psutil
import setproctitle
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
from sglang.srt.utils import (
    configure_logger,
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


class DetokenizerManager:
    """DetokenizerManager is a process that detokenizes the token ids."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        worker_id: int = 0,
    ):
        # Store worker ID for logging
        self.worker_id = worker_id

        # Init inter-process communication
        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
        )
        self.send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_ipc_name, False
        )

        if server_args.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )

        self.decode_status = LimitedCapacityDict(capacity=DETOKENIZER_MAX_STATES)
        self.is_dummy = server_args.load_format == "dummy"

        # Performance monitoring
        self.server_args = server_args
        self.performance_stats = {
            "total_requests": 0,
            "total_tokens_processed": 0,
            "total_processing_time": 0.0,
            "queue_size_history": [],
            "processing_latency_history": [],
            "last_log_time": 0.0,
            "log_interval": getattr(
                server_args, "detokenizer_log_interval", 100
            ),  # Log every 100 requests
            "enable_detokenizer_logging": getattr(
                server_args, "enable_detokenizer_logging", False
            ),
        }

        self._request_dispatcher = TypeBasedDispatcher(
            [
                (BatchEmbeddingOut, self.handle_batch_embedding_out),
                (BatchTokenIDOut, self.handle_batch_token_id_out),
                (BatchMultimodalDecodeReq, self.handle_multimodal_decode_req),
            ]
        )

        # Initialize performance monitoring
        if self.performance_stats["enable_detokenizer_logging"]:
            logger.info(
                f"🔌 Worker {self.worker_id} - 🔍 Detokenizer performance monitoring enabled"
            )
            logger.info(
                f"🔌 Worker {self.worker_id} - 📊 Log interval: every {self.performance_stats['log_interval']} requests"
            )

        # Log worker startup
        logger.info(
            f"🔌 Worker {self.worker_id} - 🚀 Detokenizer worker started successfully"
        )
        logger.info(
            f"🔌 Worker {self.worker_id} - 📍 IPC endpoint: {port_args.detokenizer_ipc_name}"
        )
        logger.info(
            f"🔌 Worker {self.worker_id} - 💾 Max states capacity: {DETOKENIZER_MAX_STATES}"
        )

    def _update_performance_stats(
        self, request_count: int, token_count: int, processing_time: float
    ):
        """Update performance statistics"""
        if not self.performance_stats["enable_detokenizer_logging"]:
            return

        self.performance_stats["total_requests"] += request_count
        self.performance_stats["total_tokens_processed"] += token_count
        self.performance_stats["total_processing_time"] += processing_time

        # Store recent history (keep last 100 entries)
        if len(self.performance_stats["processing_latency_history"]) >= 100:
            self.performance_stats["processing_latency_history"].pop(0)
        self.performance_stats["processing_latency_history"].append(processing_time)

        # Log performance stats periodically
        if (
            self.performance_stats["total_requests"]
            % self.performance_stats["log_interval"]
            == 0
            and self.performance_stats["total_requests"] > 0
        ):
            self._log_performance_stats()

    def _log_performance_stats(self):
        """Log comprehensive performance statistics"""
        if not self.performance_stats["enable_detokenizer_logging"]:
            return

        total_reqs = self.performance_stats["total_requests"]
        total_tokens = self.performance_stats["total_tokens_processed"]
        total_time = self.performance_stats["total_processing_time"]

        # Calculate metrics
        avg_latency = total_time / total_reqs if total_reqs > 0 else 0
        throughput = total_tokens / total_time if total_time > 0 else 0
        queue_size = len(self.decode_status)

        # Calculate recent latency statistics
        recent_latencies = self.performance_stats["processing_latency_history"][
            -10:
        ]  # Last 10 requests
        avg_recent_latency = (
            sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0
        )
        max_recent_latency = max(recent_latencies) if recent_latencies else 0

        # Determine bottleneck status
        bottleneck_status = "🟢 Healthy"
        if avg_recent_latency > 0.1:  # > 100ms
            bottleneck_status = "🔴 Bottleneck"
        elif avg_recent_latency > 0.05:  # > 50ms
            bottleneck_status = "🟡 Warning"

        # Log comprehensive stats
        logger.info(
            f"🔌 Worker {self.worker_id} - 📊 Detokenizer Performance Stats (last {self.performance_stats['log_interval']} requests):"
        )
        logger.info(
            f"   {bottleneck_status} - Avg Latency: {avg_recent_latency*1000:.1f}ms, Max: {max_recent_latency*1000:.1f}ms"
        )
        logger.info(
            f"   📈 Throughput: {throughput:.1f} tokens/sec, Queue Size: {queue_size}"
        )
        logger.info(
            f"   📊 Total: {total_reqs} requests, {total_tokens} tokens, {total_time:.2f}s"
        )

        # Add worker summary for better visibility
        logger.info(
            f"🔌 Worker {self.worker_id} - 📋 Summary: Queue={queue_size}, "
            f"Throughput={throughput:.0f} tokens/sec, "
            f"Latency={avg_recent_latency*1000:.0f}ms"
        )

        # Log queue size warning if high
        if queue_size > 1000:
            logger.warning(
                f"🔌 Worker {self.worker_id} - ⚠️  High detokenizer queue size: {queue_size} requests - potential bottleneck!"
            )
        elif queue_size > 500:
            logger.info(
                f"🔌 Worker {self.worker_id} - 📋 Moderate detokenizer queue size: {queue_size} requests"
            )

        # Store queue size history for trend analysis
        if len(self.performance_stats["queue_size_history"]) >= 100:
            self.performance_stats["queue_size_history"].pop(0)
        self.performance_stats["queue_size_history"].append(queue_size)

        # Log queue size trends
        if len(self.performance_stats["queue_size_history"]) >= 10:
            recent_queue_sizes = self.performance_stats["queue_size_history"][-10:]
            avg_queue_size = sum(recent_queue_sizes) / len(recent_queue_sizes)
            if queue_size > avg_queue_size * 1.5:  # 50% increase
                logger.warning(
                    f"🔌 Worker {self.worker_id} - 📈 Queue size increasing: {queue_size} (avg: {avg_queue_size:.1f}) - monitor for bottlenecks"
                )

    def event_loop(self):
        """The event loop that handles requests"""
        while True:
            start_time = time.time()
            recv_obj = self.recv_from_scheduler.recv_pyobj()

            # Process the request
            output = self._request_dispatcher(recv_obj)
            self.send_to_tokenizer.send_pyobj(output)

            # Update performance stats
            processing_time = time.time() - start_time
            request_count = 1
            token_count = self._count_tokens_in_request(recv_obj)
            self._update_performance_stats(request_count, token_count, processing_time)

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
        manager.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(
            f"🔌 Worker {worker_id} - DetokenizerManager hit an exception: {traceback}"
        )
        parent_process.send_signal(signal.SIGQUIT)
