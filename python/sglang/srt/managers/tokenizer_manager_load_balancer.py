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
"""Load balancer for multiple TokenizerManager workers."""

import logging
import random
import time
from typing import List, Optional

import zmq

from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_zmq_socket

logger = logging.getLogger(__name__)


class TokenizerManagerLoadBalancer:
    """Load balancer for multiple TokenizerManager workers."""

    def __init__(self, server_args: ServerArgs, port_args_list: List[PortArgs]):
        self.server_args = server_args
        self.port_args_list = port_args_list
        self.num_workers = len(port_args_list)
        self.load_balance_method = server_args.tokenizer_manager_worker_load_balance

        # Initialize ZMQ sockets for each worker
        self.worker_sockets = []
        context = zmq.Context(2)

        for i, port_args in enumerate(port_args_list):
            socket = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_manager_ipc_name, False
            )
            self.worker_sockets.append(socket)
            logger.info(
                f"🔀 TokenizerManager Load Balancer: Connected to worker {i} at {port_args.tokenizer_manager_ipc_name}"
            )

        # Load balancing state
        self.current_worker_index = 0
        self.worker_connection_counts = [0] * self.num_workers
        self.last_worker_selection = time.time()

        logger.info(
            f"🔀 TokenizerManager Load Balancer: Initialized with {self.num_workers} workers, "
            f"method: {self.load_balance_method}"
        )

    def send_pyobj(self, obj):
        """Send object to a worker using the configured load balancing strategy."""
        worker_index = self._select_worker()

        try:
            self.worker_sockets[worker_index].send_pyobj(obj)
            self._update_worker_stats(worker_index)

            if self.server_args.enable_detokenizer_worker_logging:
                logger.debug(
                    f"🔀 TokenizerManager Load Balancer: Sent to worker {worker_index}"
                )

        except Exception as e:
            logger.error(
                f"🔀 TokenizerManager Load Balancer: Failed to send to worker {worker_index}: {e}"
            )
            # Try to send to another worker as fallback
            self._send_with_fallback(obj, worker_index)

    def _select_worker(self) -> int:
        """Select worker based on load balancing strategy."""
        if self.load_balance_method == "round_robin":
            return self._round_robin_selection()
        elif self.load_balance_method == "least_connections":
            return self._least_connections_selection()
        elif self.load_balance_method == "random":
            return self._random_selection()
        else:
            logger.warning(
                f"🔀 TokenizerManager Load Balancer: Unknown method '{self.load_balance_method}', "
                f"falling back to round_robin"
            )
            return self._round_robin_selection()

    def _round_robin_selection(self) -> int:
        """Round-robin worker selection."""
        worker_index = self.current_worker_index
        self.current_worker_index = (self.current_worker_index + 1) % self.num_workers
        return worker_index

    def _least_connections_selection(self) -> int:
        """Select worker with least active connections."""
        min_connections = min(self.worker_connection_counts)
        candidates = [
            i
            for i, count in enumerate(self.worker_connection_counts)
            if count == min_connections
        ]

        if len(candidates) == 1:
            return candidates[0]
        else:
            # If multiple workers have same count, use round-robin among them
            return candidates[self.current_worker_index % len(candidates)]

    def _random_selection(self) -> int:
        """Random worker selection."""
        return random.randint(0, self.num_workers - 1)

    def _update_worker_stats(self, worker_index: int):
        """Update worker connection statistics."""
        self.worker_connection_counts[worker_index] += 1

        # Reset counters periodically to prevent overflow
        current_time = time.time()
        if current_time - self.last_worker_selection > 3600:  # Every hour
            self.worker_connection_counts = [
                count // 2 for count in self.worker_connection_counts
            ]
            self.last_worker_selection = current_time

    def _send_with_fallback(self, obj, failed_worker_index: int):
        """Send to another worker if the primary worker fails."""
        # Try to find another available worker
        for i in range(self.num_workers):
            if i != failed_worker_index:
                try:
                    self.worker_sockets[i].send_pyobj(obj)
                    self._update_worker_stats(i)
                    logger.info(
                        f"🔀 TokenizerManager Load Balancer: Fallback successful - sent to worker {i}"
                    )
                    return
                except Exception as e:
                    logger.error(
                        f"🔀 TokenizerManager Load Balancer: Fallback to worker {i} also failed: {e}"
                    )

        # If all workers fail, raise the exception
        raise RuntimeError(
            f"🔀 TokenizerManager Load Balancer: All {self.num_workers} workers failed"
        )

    def get_worker_stats(self) -> dict:
        """Get current worker statistics."""
        return {
            "num_workers": self.num_workers,
            "load_balance_method": self.load_balance_method,
            "worker_connection_counts": self.worker_connection_counts.copy(),
            "current_worker_index": self.current_worker_index,
        }

    def close(self):
        """Close all worker connections."""
        for i, socket in enumerate(self.worker_sockets):
            try:
                socket.close()
                logger.info(
                    f"🔀 TokenizerManager Load Balancer: Closed connection to worker {i}"
                )
            except Exception as e:
                logger.error(
                    f"🔀 TokenizerManager Load Balancer: Error closing worker {i}: {e}"
                )

        self.worker_sockets.clear()
        logger.info("🔀 TokenizerManager Load Balancer: All connections closed")
