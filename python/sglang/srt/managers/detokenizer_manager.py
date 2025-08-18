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
from typing import Dict, List, Set, Union

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
        process_id: int = 0,
    ):
        self.process_id = process_id
        self.port_args = port_args  # Store port_args for later use

        # Init inter-process communication
        context = zmq.Context(2)

        # For multi-process mode, receive from coordinator; otherwise from scheduler directly
        if server_args.detokenizer_processes > 1:
            logger.info(
                f"🔌 DetokenizerManager {process_id}: Setting up multi-process mode"
            )
            logger.info(
                f"  Port args detokenizer_ipc_names: {port_args.detokenizer_ipc_names}"
            )
            logger.info(
                f"  Process {process_id} listening on: {port_args.detokenizer_ipc_names[process_id]}"
            )
            try:
                logger.info(
                    f"🔌 DetokenizerManager {process_id}: Attempting to bind to {port_args.detokenizer_ipc_names[process_id]}"
                )
                self.recv_from_coordinator = get_zmq_socket(
                    context, zmq.PULL, port_args.detokenizer_ipc_names[process_id], True
                )
                if getattr(
                    server_args, "enable_detokenizer_coordination_logging", False
                ):
                    logger.info(
                        f"✅ DetokenizerManager {process_id}: Successfully bound to {port_args.detokenizer_ipc_names[process_id]}"
                    )
            except Exception as e:
                logger.error(
                    f"❌ DetokenizerManager {process_id}: Failed to bind to {port_args.detokenizer_ipc_names[process_id]}: {e}"
                )
                raise
            # For HTTP-based communication, no external load balancer connection is needed
            # The response goes back through the same HTTP connection
            self.send_to_load_balancer = None
            # Keep backward compatibility socket for single-process mode
            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
            if getattr(server_args, "enable_detokenizer_coordination_logging", False):
                logger.info(
                    f"✅ DetokenizerManager {process_id}: Multi-process setup complete"
                )
        else:
            # Single process mode - receive from scheduler, send to tokenizer
            if getattr(server_args, "enable_detokenizer_coordination_logging", False):
                logger.info(
                    f"🔌 DetokenizerManager {process_id}: Setting up single-process mode"
                )
                logger.info(f"  Listening on: {port_args.detokenizer_ipc_name}")
            self.recv_from_scheduler = get_zmq_socket(
                context, zmq.PULL, port_args.detokenizer_ipc_name, True
            )
            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
            self.recv_from_coordinator = None
            self.send_to_load_balancer = None
            if getattr(server_args, "enable_detokenizer_coordination_logging", False):
                logger.info(
                    f"✅ DetokenizerManager {process_id}: Single-process setup complete"
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

        # Track active requests for multi-process mode
        self.active_requests: Set[str] = set()

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
            "process_id": process_id,
        }

        # Enhanced queue monitoring
        self.queue_monitoring = {
            "input_queue_size": 0,  # Current input queue size
            "last_log_time": time.time(),
            "log_interval": 10,  # Log every 10 seconds
        }

        # Request processing tracking
        self.request_processing_stats = {
            "requests_in_progress": 0,
            "requests_completed": 0,
            "requests_failed": 0,
            "avg_processing_time": 0.0,
            "min_processing_time": float("inf"),
            "max_processing_time": 0.0,
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
            if getattr(server_args, "enable_detokenizer_coordination_logging", False):
                logger.info(
                    f"🔍 Detokenizer {process_id} performance monitoring enabled"
                )
                logger.info(
                    f"📊 Log interval: every {self.performance_stats['log_interval']} requests"
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

        # Log comprehensive stats in a single line
        logger.info(
            f"📊 Detokenizer {self.process_id}: {bottleneck_status} | "
            f"Latency: {avg_recent_latency*1000:.1f}ms avg, {max_recent_latency*1000:.1f}ms max | "
            f"Throughput: {throughput:.1f} tokens/sec | "
            f"Queue: {queue_size} | "
            f"Total: {total_reqs} reqs, {total_tokens} tokens, {total_time:.2f}s"
        )

        # Log queue size warning if high
        if queue_size > 1000:
            logger.warning(
                f"⚠️  High detokenizer queue size: {queue_size} requests - potential bottleneck!"
            )
        elif queue_size > 500:
            logger.info(f"📋 Moderate detokenizer queue size: {queue_size} requests")

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
                    f"📈 Queue size increasing: {queue_size} (avg: {avg_queue_size:.1f}) - monitor for bottlenecks"
                )

    def _track_request(self, request_id: str):
        """Track that this request is being processed by this DetokenizerManager."""
        self.active_requests.add(request_id)
        logger.debug(
            f"DetokenizerManager {self.process_id} now tracking request {request_id}"
        )

    def _untrack_request(self, request_id: str):
        """Stop tracking a completed request."""
        if request_id in self.active_requests:
            self.active_requests.remove(request_id)
            logger.debug(
                f"DetokenizerManager {self.process_id} stopped tracking request {request_id}"
            )

    def _get_request_id(self, request) -> str:
        """Extract request ID from the request object."""
        if hasattr(request, "rids") and len(request.rids) > 0:
            # For batch requests, use the first request ID as the batch identifier
            return request.rids[0]
        elif hasattr(request, "rid"):
            return request.rid
        else:
            # Fallback: generate a unique identifier
            return str(id(request))

    def _send_result(self, result):
        """Send result to appropriate destination based on mode."""
        if self.server_args.detokenizer_processes > 1:
            # Multi-process mode: send directly to tokenizer
            # For HTTP-based communication, the response goes back through the same HTTP connection
            self.send_to_tokenizer.send_pyobj(result)
        else:
            # Single process mode: send to tokenizer
            self.send_to_tokenizer.send_pyobj(result)

    def event_loop(self):
        """Main event loop that handles both single and multi-process modes."""
        if getattr(self.server_args, "enable_detokenizer_coordination_logging", False):
            logger.info(f"DetokenizerManager {self.process_id} event loop started")

        # Debug: Check if port_args is available
        if getattr(self.server_args, "enable_detokenizer_coordination_logging", False):
            logger.info(
                f"🔍 DetokenizerManager {self.process_id}: Checking attributes..."
            )
            logger.info(f"  Has port_args: {hasattr(self, 'port_args')}")
            logger.info(f"  Has server_args: {hasattr(self, 'server_args')}")
            if hasattr(self, "port_args"):
                logger.info(f"  port_args type: {type(self.port_args)}")
                logger.info(
                    f"  port_args detokenizer_ipc_names: {getattr(self.port_args, 'detokenizer_ipc_names', 'NOT_FOUND')}"
                )

        if self.server_args.detokenizer_processes > 1:
            if getattr(
                self.server_args, "enable_detokenizer_coordination_logging", False
            ):
                logger.info(
                    f"🔍 DetokenizerManager {self.process_id} waiting for requests from coordinator at {self.port_args.detokenizer_ipc_names[self.process_id]}"
                )
        else:
            if getattr(
                self.server_args, "enable_detokenizer_coordination_logging", False
            ):
                logger.info(
                    f"🔍 DetokenizerManager {self.process_id} waiting for requests from scheduler at {self.port_args.detokenizer_ipc_name}"
                )

        while True:
            try:
                # Receive request based on mode
                if self.server_args.detokenizer_processes > 1:
                    # Multi-process mode: receive from coordinator
                    request = self.recv_from_coordinator.recv_pyobj()
                    if getattr(
                        self.server_args,
                        "enable_detokenizer_coordination_logging",
                        False,
                    ):
                        logger.info(
                            f"✅ DetokenizerManager {self.process_id} received request from coordinator: {type(request).__name__}"
                        )
                else:
                    # Single process mode: receive from scheduler
                    request = self.recv_from_scheduler.recv_pyobj()
                    if getattr(
                        self.server_args,
                        "enable_detokenizer_coordination_logging",
                        False,
                    ):
                        logger.info(
                            f"✅ DetokenizerManager {self.process_id} received request from scheduler: {type(request).__name__}"
                        )

                # Handle shutdown command (multi-process mode)
                if isinstance(request, dict) and request.get("command") == "shutdown":
                    if getattr(
                        self.server_args,
                        "enable_detokenizer_coordination_logging",
                        False,
                    ):
                        logger.info(
                            f"DetokenizerManager {self.process_id} received shutdown command"
                        )
                    break

                # Extract request ID for tracking (multi-process mode)
                if self.server_args.detokenizer_processes > 1:
                    request_id = self._get_request_id(request)
                    self._track_request(request_id)

                # Process the request with timing
                start_time = time.time()
                try:
                    result = self._process_request(request)
                    processing_time = time.time() - start_time
                    success = True
                except Exception as e:
                    processing_time = time.time() - start_time
                    success = False
                    logger.error(f"Failed to process request: {e}")
                    result = None

                # Update performance stats
                request_count = len(getattr(request, "rids", [1]))
                token_count = len(getattr(result, "output_ids", [])) if result else 0
                self._update_performance_stats(
                    request_count, token_count, processing_time
                )

                # Send result to appropriate destination (only if successful)
                if success and result:
                    self._send_result(result)

                    # Untrack completed requests (multi-process mode)
                    if self.server_args.detokenizer_processes > 1 and hasattr(
                        result, "rids"
                    ):
                        for rid in result.rids:
                            self._untrack_request(rid)

                # Monitor queue sizes and log statistics
                self._monitor_queue_sizes()

                logger.debug(
                    f"DetokenizerManager {self.process_id} processed request in {processing_time:.3f}s"
                )

            except Exception as e:
                logger.error(f"Error in DetokenizerManager {self.process_id}: {e}")
                break

        if getattr(self.server_args, "enable_detokenizer_coordination_logging", False):
            logger.info(f"DetokenizerManager {self.process_id} event loop ended")

    def get_stats(self) -> Dict[str, any]:
        """Get comprehensive performance statistics for this DetokenizerManager."""
        stats = self.performance_stats.copy()
        if self.server_args.detokenizer_processes > 1:
            stats["active_requests_count"] = len(self.active_requests)
            stats["active_requests"] = list(self.active_requests)

        # Calculate derived metrics
        if stats["total_requests"] > 0:
            stats["avg_processing_time"] = (
                stats["total_processing_time"] / stats["total_requests"]
            )
        if stats["total_processing_time"] > 0:
            stats["throughput_tokens_per_sec"] = (
                stats["total_tokens_processed"] / stats["total_processing_time"]
            )

        # Add queue monitoring stats
        stats["queue_monitoring"] = {
            "current_input_queue_size": self.queue_monitoring["input_queue_size"],
            "current_processing_queue_size": 0,  # No longer tracking processing queue size
            "max_input_queue_size": 0,
            "max_processing_queue_size": 0,
            "input_queue_history": self.performance_stats["queue_size_history"][
                -10:
            ],  # Last 10 entries
            "processing_queue_history": [],  # No longer tracking processing queue history
        }

        # Add request processing stats
        stats["request_processing"] = self.request_processing_stats.copy()

        # Add queue efficiency metrics
        total_requests = stats["total_requests"]
        if total_requests > 0:
            stats["queue_efficiency"] = {
                "input_queue_utilization": self.queue_monitoring["input_queue_size"]
                / total_requests
                * 100,
                "processing_queue_utilization": 0,  # No longer tracking processing queue utilization
                "queue_health": self._get_queue_health_status(),
            }

        return stats

    def _get_queue_health_status(self) -> str:
        """Get a human-readable queue health status."""
        total_queue = self.queue_monitoring["input_queue_size"]

        if total_queue == 0:
            return "🟢 Empty"
        elif total_queue <= 5:
            return "🟢 Healthy"
        elif total_queue <= 15:
            return "🟡 Moderate"
        elif total_queue <= 30:
            return "🟠 High"
        else:
            return "🔴 Critical"

    def _monitor_queue_sizes(self):
        """Monitor and track queue sizes over time."""
        current_time = time.time()
        if (
            current_time - self.queue_monitoring["last_log_time"]
            < self.queue_monitoring["log_interval"]
        ):
            return

        # Update current queue sizes
        self.queue_monitoring["input_queue_size"] = len(self.active_requests)

        # Log queue status if monitoring is enabled
        if self.performance_stats["enable_detokenizer_logging"]:
            self._log_queue_status()

        # Update last monitor time
        self.queue_monitoring["last_log_time"] = current_time

    def _log_queue_status(self):
        """Log current queue status and trends."""
        input_queue = self.queue_monitoring["input_queue_size"]

        # Determine queue health status
        if input_queue == 0:
            status = "🟢 Empty"
        elif input_queue <= 5:
            status = "🟢 Healthy"
        elif input_queue <= 15:
            status = "🟡 Moderate"
        elif input_queue <= 30:
            status = "🟠 High"
        else:
            status = "🔴 Critical"

        # Calculate queue trends
        input_trend = "→ Stable"

        if len(self.performance_stats["queue_size_history"]) >= 2:
            prev_input = self.performance_stats["queue_size_history"][-2]
            curr_input = self.performance_stats["queue_size_history"][-1]
            if curr_input > prev_input:
                input_trend = "↗️ Increasing"
            elif curr_input < prev_input:
                input_trend = "↘️ Decreasing"

        logger.info(
            f"📋 DetokenizerManager {self.process_id}: {status} | "
            f"Input: {input_queue} ({input_trend}) | "
            f"Processing: 0 (→ Stable) | "
            f"Total: {input_queue} | "
            f"Active: {list(self.active_requests)[:5]}{'...' if len(self.active_requests) > 5 else ''}"
        )

    def _update_request_processing_stats(
        self, processing_time: float, success: bool = True
    ):
        """Update request processing statistics."""
        if success:
            self.request_processing_stats["requests_completed"] += 1
        else:
            self.request_processing_stats["requests_failed"] += 1

        # Update timing statistics
        if processing_time < self.request_processing_stats["min_processing_time"]:
            self.request_processing_stats["min_processing_time"] = processing_time
        if processing_time > self.request_processing_stats["max_processing_time"]:
            self.request_processing_stats["max_processing_time"] = processing_time

        # Update average processing time
        total_completed = self.request_processing_stats["requests_completed"]
        if total_completed > 0:
            current_avg = self.request_processing_stats["avg_processing_time"]
            self.request_processing_stats["avg_processing_time"] = (
                current_avg * (total_completed - 1) + processing_time
            ) / total_completed

    def _log_detailed_performance_stats(self):
        """Log detailed performance and queue statistics."""
        current_time = time.time()
        if (
            current_time - self.performance_stats["last_log_time"]
            < self.performance_stats["log_interval"]
        ):
            return

        # Calculate queue efficiency metrics
        total_requests = self.performance_stats["total_requests"]
        avg_processing_time = (
            self.performance_stats["total_processing_time"] / total_requests
            if total_requests > 0
            else 0
        )
        throughput = (
            self.performance_stats["total_tokens_processed"]
            / self.performance_stats["total_processing_time"]
            if self.performance_stats["total_processing_time"] > 0
            else 0
        )

        # Calculate queue utilization
        input_queue_utilization = (
            self.queue_monitoring["input_queue_size"] / max(1, total_requests) * 100
        )
        processing_queue_utilization = (
            0  # No longer tracking processing queue utilization
        )

        # Log comprehensive performance report
        logger.info("=" * 80)
        logger.info(f"🚀 DETOKENIZERMANAGER {self.process_id} PERFORMANCE REPORT")
        logger.info("=" * 80)
        logger.info(f"📊 Processing Statistics:")
        logger.info(f"  Total Requests: {total_requests}")
        logger.info(
            f"  Total Tokens: {self.performance_stats['total_tokens_processed']}"
        )
        logger.info(
            f"  Total Time: {self.performance_stats['total_processing_time']:.2f}s"
        )
        logger.info(f"  Average Processing Time: {avg_processing_time*1000:.1f}ms")
        logger.info(f"  Throughput: {throughput:.1f} tokens/sec")

        logger.info(f"📋 Queue Statistics:")
        logger.info(
            f"  Input Queue: {self.queue_monitoring['input_queue_size']} (Peak: 0)"
        )  # Peak is 0 as per new structure
        logger.info(
            f"  Processing Queue: 0 (Peak: 0)"
        )  # No longer tracking processing queue
        logger.info(f"  Input Queue Utilization: {input_queue_utilization:.1f}%")
        logger.info(
            f"  Processing Queue Utilization: 0.0%"
        )  # No longer tracking processing queue utilization

        logger.info(f"⚡ Request Processing Stats:")
        logger.info(
            f"  In Progress: {self.request_processing_stats['requests_in_progress']}"
        )
        logger.info(
            f"  Completed: {self.request_processing_stats['requests_completed']}"
        )
        logger.info(f"  Failed: {self.request_processing_stats['requests_failed']}")
        logger.info(
            f"  Min Processing Time: {self.request_processing_stats['min_processing_time']*1000:.1f}ms"
        )
        logger.info(
            f"  Max Processing Time: {self.request_processing_stats['max_processing_time']*1000:.1f}ms"
        )
        logger.info(
            f"  Avg Processing Time: {self.request_processing_stats['avg_processing_time']*1000:.1f}ms"
        )

        # Log recent queue history trends
        if len(self.performance_stats["queue_size_history"]) >= 3:
            recent_input_sizes = [
                entry for entry in self.performance_stats["queue_size_history"][-3:]
            ]
            input_trend = (
                "↗️ Increasing"
                if recent_input_sizes[-1] > recent_input_sizes[0]
                else (
                    "↘️ Decreasing"
                    if recent_input_sizes[-1] < recent_input_sizes[0]
                    else "→ Stable"
                )
            )
            logger.info(
                f"📈 Recent Input Queue Trend: {input_trend} {recent_input_sizes}"
            )

        if len(self.performance_stats["queue_size_history"]) >= 3:
            recent_processing_sizes = []  # No longer tracking processing queue history
            processing_trend = "→ Stable"
            logger.info(
                f"📈 Recent Processing Queue Trend: {processing_trend} {recent_processing_sizes}"
            )

        logger.info("=" * 80)

        # Update last log time
        self.performance_stats["last_log_time"] = current_time

    def _process_request(self, request):
        """Process a request using the request dispatcher."""
        return self._request_dispatcher(request)

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


class DetokenizerCoordinator:
    """Coordinates multiple DetokenizerManager processes for load balancing only."""

    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        self.server_args = server_args
        self.port_args = port_args
        self.detokenizer_processes = server_args.detokenizer_processes
        self.load_balance_policy = server_args.detokenizer_load_balance_policy

        # ZMQ sockets for communication
        self.context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            self.context, zmq.PULL, port_args.detokenizer_coordinator_ipc_name, True
        )

        # Create sockets to each DetokenizerManager process
        self.detokenizer_sockets = []
        logger.info(
            f"🔌 DetokenizerCoordinator: Creating sockets to {len(port_args.detokenizer_ipc_names)} DetokenizerManager processes"
        )
        for i, ipc_name in enumerate(port_args.detokenizer_ipc_names):
            logger.info(f"  Process {i}: {ipc_name}")
            socket = get_zmq_socket(self.context, zmq.PUSH, ipc_name, False)
            self.detokenizer_sockets.append(socket)
        logger.info(
            f"✅ DetokenizerCoordinator: Created {len(self.detokenizer_sockets)} sockets"
        )

        # Request affinity tracking - maps request_id to assigned DetokenizerManager
        self.request_affinity: Dict[str, int] = {}

        # Load balancing state
        self.current_detokenizer = 0  # For round-robin
        self.detokenizer_loads = [0] * self.detokenizer_processes  # For least-loaded

        # Enhanced monitoring and metrics
        self.monitoring_stats = {
            "total_requests_received": 0,
            "total_requests_assigned": 0,
            "last_log_time": time.time(),
            "log_interval": 10,  # Log every 10 seconds
            "load_balance_decisions": {
                "affinity_hit": 0,
                "round_robin": 0,
                "least_loaded": 0,
                "weighted": 0,
            },
            "queue_size_history": [],
            "load_distribution_history": [],
        }

        # Performance tracking
        self.request_timing = {}  # Track request processing time
        self.last_queue_sizes = [0] * self.detokenizer_processes

        logger.info(
            f"DetokenizerCoordinator initialized with {self.detokenizer_processes} processes"
        )
        logger.info(f"Load balancing policy: {self.load_balance_policy}")
        logger.info(
            f"Monitoring enabled with {self.detokenizer_processes} detokenizer processes"
        )

    def _get_request_id(self, request) -> str:
        """Extract request ID from the request object."""
        if hasattr(request, "rids") and len(request.rids) > 0:
            # For batch requests, use the first request ID as the batch identifier
            return request.rids[0]
        elif hasattr(request, "rid"):
            return request.rid
        else:
            # Fallback: generate a unique identifier
            return str(id(request))

    def _get_or_assign_detokenizer(self, request_id: str, request_size: int) -> int:
        """Get the assigned DetokenizerManager for a request, or assign one if new."""
        if request_id in self.request_affinity:
            # Return existing assignment to maintain affinity
            self.monitoring_stats["load_balance_decisions"]["affinity_hit"] += 1
            return self.request_affinity[request_id]
        else:
            # Assign new DetokenizerManager based on load balancing policy
            assigned_detokenizer = self._select_detokenizer_for_new_request(
                request_size
            )
            self.request_affinity[request_id] = assigned_detokenizer

            # Update load tracking
            self.detokenizer_loads[assigned_detokenizer] += request_size

            # Track load balancing decision
            if self.load_balance_policy == "round_robin":
                self.monitoring_stats["load_balance_decisions"]["round_robin"] += 1
            elif self.load_balance_policy == "least_loaded":
                self.monitoring_stats["load_balance_decisions"]["least_loaded"] += 1
            elif self.load_balance_policy == "weighted":
                self.monitoring_stats["load_balance_decisions"]["weighted"] += 1

            logger.debug(
                f"Assigned request {request_id} to DetokenizerManager {assigned_detokenizer}"
            )
            return assigned_detokenizer

    def _select_detokenizer_for_new_request(self, request_size: int) -> int:
        """Select which DetokenizerManager process to use for a new request."""
        if self.load_balance_policy == "round_robin":
            selected = self.current_detokenizer
            self.current_detokenizer = (
                self.current_detokenizer + 1
            ) % self.detokenizer_processes
            return selected
        elif self.load_balance_policy == "least_loaded":
            return min(
                range(self.detokenizer_processes),
                key=lambda i: self.detokenizer_loads[i],
            )
        elif self.load_balance_policy == "weighted":
            # Consider both load and request size
            weighted_loads = [
                self.detokenizer_loads[i] + (request_size * 0.1)
                for i in range(self.detokenizer_processes)
            ]
            return min(
                range(self.detokenizer_processes), key=lambda i: weighted_loads[i]
            )
        else:
            return 0

    def _cleanup_completed_requests(self, completed_request_ids: List[str]):
        """Clean up completed requests from affinity tracking and update loads."""
        for request_id in completed_request_ids:
            if request_id in self.request_affinity:
                detokenizer_idx = self.request_affinity[request_id]
                # Estimate load reduction
                self.detokenizer_loads[detokenizer_idx] = max(
                    0, self.detokenizer_loads[detokenizer_idx] - 1
                )
                del self.request_affinity[request_id]

                logger.debug(
                    f"Cleaned up completed request {request_id} from DetokenizerManager {detokenizer_idx}"
                )

    def _log_monitoring_stats(self):
        """Log basic monitoring statistics."""
        current_time = time.time()
        if (
            current_time - self.monitoring_stats["last_log_time"]
            < self.monitoring_stats["log_interval"]
        ):
            return

        # Only log if coordination logging is enabled
        if not getattr(
            self.server_args, "enable_detokenizer_coordination_logging", False
        ):
            return

        # Calculate current queue sizes (requests waiting to be processed)
        current_queue_sizes = []
        for i in range(self.detokenizer_processes):
            queue_size = max(0, self.detokenizer_loads[i])
            current_queue_sizes.append(queue_size)

        # Log basic stats
        logger.info("=" * 60)
        logger.info("🔍 DETOKENIZER COORDINATOR STATUS")
        logger.info("=" * 60)
        logger.info(f"📊 Load Balancing Policy: {self.load_balance_policy}")
        logger.info(
            f"📈 Total Requests: Received={self.monitoring_stats['total_requests_received']}, Assigned={self.monitoring_stats['total_requests_assigned']}"
        )
        logger.info(f"🔄 Current Load Distribution: {self.detokenizer_loads}")
        logger.info(f"📋 Current Queue Sizes: {current_queue_sizes}")
        logger.info(f"🎯 Active Affinity Mappings: {len(self.request_affinity)}")

        # Log per-process details
        for i in range(self.detokenizer_processes):
            load = self.detokenizer_loads[i]
            queue_size = current_queue_sizes[i]
            logger.info(f"  Process {i}: Load={load}, Queue={queue_size}")

        logger.info("=" * 60)

        # Update last log time
        self.monitoring_stats["last_log_time"] = current_time

    def event_loop(self):
        """Main event loop for the coordinator - only handles request distribution."""
        if getattr(self.server_args, "enable_detokenizer_coordination_logging", False):
            logger.info("DetokenizerCoordinator event loop started")
            logger.info(
                f"Waiting for requests from scheduler at {self.port_args.detokenizer_coordinator_ipc_name}"
            )
            logger.info(
                f"Ready to forward to {len(self.detokenizer_sockets)} DetokenizerManager processes"
            )

        while True:
            try:
                # Receive request from scheduler
                request = self.recv_from_scheduler.recv_pyobj()
                if getattr(
                    self.server_args, "enable_detokenizer_coordination_logging", False
                ):
                    logger.info(
                        f"✅ DetokenizerCoordinator received request: {type(request).__name__}"
                    )

                # Update monitoring stats
                self.monitoring_stats["total_requests_received"] += 1

                # Extract request identifier for affinity tracking
                request_id = self._get_request_id(request)
                request_size = len(getattr(request, "rids", [1]))
                if getattr(
                    self.server_args, "enable_detokenizer_coordination_logging", False
                ):
                    logger.info(f"📋 Request ID: {request_id}, Size: {request_size}")

                # Get or assign DetokenizerManager for this request
                detokenizer_idx = self._get_or_assign_detokenizer(
                    request_id, request_size
                )

                # Update monitoring stats
                self.monitoring_stats["total_requests_assigned"] += 1

                # Forward request to assigned DetokenizerManager
                if getattr(
                    self.server_args, "enable_detokenizer_coordination_logging", False
                ):
                    logger.info(
                        f"📤 Forwarding request {request_id} to DetokenizerManager {detokenizer_idx}"
                    )
                    logger.info(
                        f"  Using socket {detokenizer_idx} to send to {self.port_args.detokenizer_ipc_names[detokenizer_idx]}"
                    )
                try:
                    self.detokenizer_sockets[detokenizer_idx].send_pyobj(request)
                    if getattr(
                        self.server_args,
                        "enable_detokenizer_coordination_logging",
                        False,
                    ):
                        logger.info(
                            f"✅ Successfully sent request {request_id} to DetokenizerManager {detokenizer_idx}"
                        )
                except Exception as e:
                    logger.error(
                        f"❌ Failed to send request {request_id} to DetokenizerManager {detokenizer_idx}: {e}"
                    )
                    import traceback

                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise

                # Log monitoring stats periodically
                self._log_monitoring_stats()

            except Exception as e:
                logger.error(f"Error in DetokenizerCoordinator: {e}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                break

        if getattr(self.server_args, "enable_detokenizer_coordination_logging", False):
            logger.info("DetokenizerCoordinator event loop ended")

    def get_stats(self) -> Dict[str, any]:
        """Get comprehensive statistics about DetokenizerManager processes and request distribution."""
        current_queue_sizes = []
        for i in range(self.detokenizer_processes):
            queue_size = max(0, self.detokenizer_loads[i])
            current_queue_sizes.append(queue_size)

        # Calculate load balancing efficiency
        total_load = sum(self.detokenizer_loads)
        avg_load = (
            total_load / self.detokenizer_processes
            if self.detokenizer_processes > 0
            else 0
        )
        load_variance = (
            sum((load - avg_load) ** 2 for load in self.detokenizer_loads)
            / self.detokenizer_processes
            if self.detokenizer_processes > 0
            else 0
        )

        return {
            "detokenizer_processes": self.detokenizer_processes,
            "current_loads": self.detokenizer_loads.copy(),
            "current_queue_sizes": current_queue_sizes,
            "request_affinity_count": len(self.request_affinity),
            "load_balance_policy": self.load_balance_policy,
            "affinity_distribution": self._get_affinity_distribution(),
            "monitoring_stats": self.monitoring_stats.copy(),
            "load_balancing_efficiency": {
                "total_load": total_load,
                "average_load": avg_load,
                "load_variance": load_variance,
                "load_std_dev": load_variance**0.5,
                "max_queue_size": (
                    max(current_queue_sizes) if current_queue_sizes else 0
                ),
                "min_queue_size": (
                    min(current_queue_sizes) if current_queue_sizes else 0
                ),
                "queue_imbalance": (
                    max(current_queue_sizes) - min(current_queue_sizes)
                    if current_queue_sizes
                    else 0
                ),
            },
            "historical_data": {
                "queue_size_history": self.monitoring_stats["queue_size_history"][
                    -10:
                ],  # Last 10 entries
                "load_distribution_history": self.monitoring_stats[
                    "load_distribution_history"
                ][
                    -10:
                ],  # Last 10 entries
            },
        }

    def _get_affinity_distribution(self) -> Dict[int, int]:
        """Get the distribution of requests across DetokenizerManager processes."""
        distribution = {i: 0 for i in range(self.detokenizer_processes)}
        for detokenizer_idx in self.request_affinity.values():
            distribution[detokenizer_idx] += 1
        return distribution

    def shutdown(self):
        """Graceful shutdown of the coordinator."""
        if getattr(self.server_args, "enable_detokenizer_coordination_logging", False):
            logger.info("Shutting down DetokenizerCoordinator...")

        # Log final statistics
        final_stats = self.get_stats()
        logger.info("Final Coordinator Statistics:")
        logger.info(
            f"  Total requests processed: {final_stats['monitoring_stats']['total_requests_received']}"
        )
        logger.info(f"  Final load distribution: {final_stats['current_loads']}")
        logger.info(f"  Final queue sizes: {final_stats['current_queue_sizes']}")

        # Send shutdown signal to all DetokenizerManager processes
        for i, socket in enumerate(self.detokenizer_sockets):
            try:
                socket.send_pyobj({"command": "shutdown", "process_id": i})
            except Exception as e:
                logger.warning(
                    f"Failed to send shutdown to DetokenizerManager {i}: {e}"
                )

        # Clean up resources
        for socket in self.detokenizer_sockets:
            socket.close()
        self.recv_from_scheduler.close()


def run_detokenizer_coordinator_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    """Run the DetokenizerCoordinator process."""
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::detokenizer_coordinator")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        coordinator = DetokenizerCoordinator(server_args, port_args)
        coordinator.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerCoordinator hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)


def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    process_id: int = 0,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::detokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        manager = DetokenizerManager(server_args, port_args, process_id)
        manager.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
