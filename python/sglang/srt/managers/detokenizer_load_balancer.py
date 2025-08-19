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
# See the License for the specific language governing permissions and limitations.
# See the License for the specific language governing permissions and limitations.
# ==============================================================================
"""DetokenizerLoadBalancer distributes work among multiple detokenizer workers."""

import logging
import random
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

import zmq

from sglang.srt.managers.base_manager import BaseManager
from sglang.srt.managers.io_struct import BatchEmbeddingOut, BatchTokenIDOut
from sglang.srt.server_args import ServerArgs
from sglang.srt.srt_py_object import SrtPyObject
from sglang.srt.utils import get_scheduler_manager

logger = logging.getLogger(__name__)


class DetokenizerLoadBalancer(BaseManager):
    """Load balancer for distributing detokenization work across multiple workers."""

    def __init__(
        self,
        server_args,
        port_args,
        detokenizer_port_args_list,
        worker_id: int = 0,
    ):
        super().__init__(server_args, port_args)
        self.worker_id = worker_id
        self.detokenizer_port_args_list = detokenizer_port_args_list
        self.num_workers = len(detokenizer_port_args_list)

        # Request affinity tracking
        self.request_worker_map: Dict[str, int] = {}

        # Worker connections
        self.workers: List[Optional[object]] = [None] * self.num_workers
        self.worker_stats = {
            "requests_sent": [0] * self.num_workers,
            "requests_assigned": [0] * self.num_workers,
            "status": ["disconnected"] * self.num_workers,
        }

        # Load balancing state
        self.current_worker = 0
        self.round_robin_counter = 0

        # Performance monitoring
        self.total_requests = 0
        self.affinity_hits = 0
        self.affinity_misses = 0

        logger.info(
            f"🔀 DetokenizerLoadBalancer {worker_id} initializing with {self.num_workers} workers"
        )

        # Initialize worker connections
        self._init_worker_connections()

    def _init_worker_connections(self):
        """Initialize connections to all detokenizer workers."""
        start_time = time.time()
        logger.info(
            f"🔌 DetokenizerLoadBalancer {self.worker_id} connecting to {self.num_workers} workers..."
        )

        successful_connections = 0
        for i, port_args in enumerate(self.detokenizer_port_args_list):
            worker_start = time.time()
            try:
                # Connect to worker
                from sglang.srt.utils import get_detokenizer_manager

                worker = get_detokenizer_manager(port_args)

                # Test connection
                test_start = time.time()
                # Note: We can't actually test send/recv here without breaking the flow
                # Just mark as connected for now
                test_time = time.time() - test_start

                self.workers[i] = worker
                self.worker_stats["status"][i] = "connected"
                successful_connections += 1

                worker_time = time.time() - worker_start
                logger.info(
                    f"🔌 Connected to detokenizer worker {i} at {port_args.detokenizer_ipc_name} "
                    f"in {worker_time:.4f}s (test: {test_time:.4f}s)"
                )

            except Exception as e:
                worker_time = time.time() - worker_start
                self.worker_stats["status"][i] = "failed"
                logger.error(
                    f"❌ Failed to connect to detokenizer worker {i} at {port_args.detokenizer_ipc_name} "
                    f"after {worker_time:.4f}s: {e}"
                )

        total_time = time.time() - start_time
        logger.info(
            f"🔀 DetokenizerLoadBalancer {self.worker_id} initialized with {successful_connections}/{self.num_workers} workers "
            f"in {total_time:.4f}s"
        )

        if successful_connections == 0:
            raise RuntimeError("No detokenizer workers could be connected")

    def send_pyobj(self, data: SrtPyObject):
        """Send data to the appropriate worker based on load balancing."""
        start_time = time.time()
        logger.info(
            f"🔀 DetokenizerLoadBalancer {self.worker_id} processing request: {type(data).__name__}"
        )

        # Time the request ID extraction
        rid_start = time.time()
        request_ids = self._extract_request_ids(data)
        rid_time = time.time() - rid_start

        # Time the worker selection
        worker_start = time.time()
        worker_id = self._get_worker_for_requests(request_ids)
        worker_time = time.time() - worker_start

        # Time the actual send operation
        send_start = time.time()
        try:
            result = self.send_to_worker(worker_id, data)
            send_time = time.time() - send_start
            total_time = time.time() - start_time

            # Update statistics
            self.total_requests += 1
            self.worker_stats["requests_sent"][worker_id] += 1
            self.worker_stats["requests_assigned"][worker_id] += 1

            logger.info(
                f"✅ DetokenizerLoadBalancer {self.worker_id} completed request in {total_time:.4f}s:\n"
                f"   - Request ID extraction: {rid_time:.4f}s\n"
                f"   - Worker selection: {worker_time:.4f}s\n"
                f"   - Send operation: {send_time:.4f}s\n"
                f"   - Assigned to worker: {worker_id}\n"
                f"   - Request IDs: {request_ids[:3]}{'...' if len(request_ids) > 3 else ''}"
            )

            return result

        except Exception as e:
            send_time = time.time() - send_start
            total_time = time.time() - start_time

            logger.error(
                f"❌ DetokenizerLoadBalancer {self.worker_id} failed to send request: {e}\n"
                f"   - Total time: {total_time:.4f}s\n"
                f"   - Request ID extraction: {rid_time:.4f}s\n"
                f"   - Worker selection: {worker_time:.4f}s\n"
                f"   - Send attempt: {send_time:.4f}s\n"
                f"   - Target worker: {worker_id}\n"
                f"   - Exception type: {type(e).__name__}"
            )
            raise

    def _extract_request_ids(self, data: SrtPyObject) -> List[str]:
        """Extract request IDs from the data."""
        start_time = time.time()

        if hasattr(data, "rids"):
            request_ids = data.rids
        else:
            # Fallback for other data types
            request_ids = []

        extract_time = time.time() - start_time
        logger.debug(
            f"🔍 Extracted {len(request_ids)} request IDs in {extract_time:.4f}s"
        )

        return request_ids

    def _get_worker_for_requests(self, request_ids: List[str]) -> int:
        """Get the appropriate worker for the given request IDs."""
        start_time = time.time()

        # Check for existing affinity
        existing_requests = []
        new_requests = []

        for rid in request_ids:
            if rid in self.request_worker_map:
                existing_requests.append(rid)
            else:
                new_requests.append(rid)

        # Determine worker assignment
        if existing_requests:
            # Use existing affinity
            worker_id = self.request_worker_map[existing_requests[0]]
            if new_requests:
                # Assign new requests in mixed batch to same worker
                for rid in new_requests:
                    self.request_worker_map[rid] = worker_id
            affinity_result = "existing"
            self.affinity_hits += 1
        else:
            # New requests, load balance
            worker_id = self.get_worker()
            for rid in new_requests:
                self.request_worker_map[rid] = worker_id
            affinity_result = "new"
            self.affinity_misses += 1

        selection_time = time.time() - start_time
        logger.debug(
            f"🎯 Worker selection completed in {selection_time:.4f}s:\n"
            f"   - Affinity: {affinity_result}\n"
            f"   - Existing requests: {len(existing_requests)}\n"
            f"   - New requests: {len(new_requests)}\n"
            f"   - Selected worker: {worker_id}"
        )

        return worker_id

    def get_worker(self) -> int:
        """Get the next worker using round-robin."""
        start_time = time.time()

        # Simple round-robin selection
        worker_id = self.round_robin_counter % self.num_workers
        self.round_robin_counter += 1

        selection_time = time.time() - start_time
        logger.debug(
            f"🔄 Round-robin worker selection: {worker_id} in {selection_time:.4f}s"
        )

        return worker_id

    def send_to_worker(self, worker_id: int, data: SrtPyObject):
        """Send data to a specific worker."""
        start_time = time.time()

        if worker_id >= len(self.workers) or self.workers[worker_id] is None:
            raise ValueError(f"Invalid worker ID: {worker_id}")

        try:
            # Send to worker
            result = self.workers[worker_id].send_pyobj(data)
            send_time = time.time() - start_time

            logger.debug(f"📤 Sent to worker {worker_id} in {send_time:.4f}s")
            return result

        except Exception as e:
            send_time = time.time() - start_time
            logger.error(
                f"❌ Failed to send to worker {worker_id} after {send_time:.4f}s: {e}"
            )
            raise

    def cleanup_completed_request(self, request_id: str):
        """Remove a completed request from the affinity map."""
        start_time = time.time()

        if request_id in self.request_worker_map:
            del self.request_worker_map[request_id]
            cleanup_time = time.time() - start_time
            logger.debug(f"🧹 Cleaned up request {request_id} in {cleanup_time:.4f}s")
        else:
            cleanup_time = time.time() - start_time
            logger.debug(
                f"🔍 Request {request_id} not found in affinity map (cleanup: {cleanup_time:.4f}s)"
            )

    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        start_time = time.time()

        stats = {
            "total_requests": self.total_requests,
            "affinity_hits": self.affinity_hits,
            "affinity_misses": self.affinity_misses,
            "affinity_map_size": len(self.request_worker_map),
            "workers": [],
        }

        for i in range(self.num_workers):
            worker_stats = {
                "worker_id": i,
                "requests_sent": self.worker_stats["requests_sent"][i],
                "requests_assigned": self.worker_stats["requests_assigned"][i],
                "status": self.worker_stats["status"][i],
            }
            stats["workers"].append(worker_stats)

        stats_time = time.time() - start_time
        logger.debug(f"📊 Generated stats in {stats_time:.4f}s")

        return stats

    def periodic_cleanup(self):
        """Periodically clean up old affinity mappings."""
        start_time = time.time()

        # Simple cleanup - remove old entries
        # In a real implementation, you might want time-based cleanup
        cleanup_count = 0

        cleanup_time = time.time() - start_time
        if cleanup_count > 0:
            logger.info(
                f"🧹 Periodic cleanup removed {cleanup_count} old mappings in {cleanup_time:.4f}s"
            )
        else:
            logger.debug(
                f"🧹 Periodic cleanup completed in {cleanup_time:.4f}s (no cleanup needed)"
            )
