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
from typing import Dict, List, Optional

import zmq

from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class DetokenizerLoadBalancer:
    """Load balancer for multiple detokenizer workers."""

    def __init__(self, server_args: ServerArgs, port_args_list: List):
        self.server_args = server_args
        self.port_args_list = port_args_list
        self.num_workers = server_args.num_detokenizer_workers
        self.load_balance_method = getattr(
            server_args, "detokenizer_load_balance_method", "round_robin"
        )

        # Initialize ZMQ sockets for each worker
        self.context = zmq.Context()
        self.worker_sockets: List[zmq.Socket] = []
        self.worker_stats: List[Dict] = []

        # Thread safety
        self.lock = threading.Lock()
        self.round_robin_counter = 0

        # Initialize stats logging
        self._last_stats_log = time.time()

        # Request affinity: map request ID to assigned worker
        self.request_worker_map: Dict[str, int] = {}

        # Cleanup tracking
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 300  # Clean up every 5 minutes

        # Initialize worker connections
        self._init_worker_connections()

        logger.info(
            f"🔀 Detokenizer load balancer initialized with {self.num_workers} workers"
        )
        logger.info(f"📊 Load balancing method: {self.load_balance_method}")
        logger.info(f"🔗 Request affinity enabled for streaming consistency")

    def _init_worker_connections(self):
        """Initialize ZMQ connections to all detokenizer workers."""
        for i, port_args in enumerate(self.port_args_list):
            try:
                # Create PUSH socket to send work to this worker
                socket = self.context.socket(zmq.PUSH)
                socket.connect(port_args.detokenizer_ipc_name)

                # Set high water mark to prevent memory issues
                socket.setsockopt(zmq.SNDHWM, 1000)

                self.worker_sockets.append(socket)

                # Initialize worker statistics
                self.worker_stats.append(
                    {
                        "worker_id": i,
                        "requests_sent": 0,
                        "last_request_time": 0,
                        "connection_status": "connected",
                        "error_count": 0,
                    }
                )

                logger.info(
                    f"🔌 Connected to detokenizer worker {i} at {port_args.detokenizer_ipc_name}"
                )

            except Exception as e:
                logger.error(f"❌ Failed to connect to detokenizer worker {i}: {e}")
                # Create a dummy socket that will fail gracefully
                self.worker_sockets.append(None)
                self.worker_stats.append(
                    {
                        "worker_id": i,
                        "requests_sent": 0,
                        "last_request_time": 0,
                        "connection_status": "failed",
                        "error_count": 1,
                    }
                )

    def _get_worker_round_robin(self) -> int:
        """Get next worker using round-robin strategy."""
        with self.lock:
            worker_id = self.round_robin_counter % self.num_workers
            self.round_robin_counter += 1
            return worker_id

    def _get_worker_least_connections(self) -> int:
        """Get worker with least number of requests sent."""
        with self.lock:
            min_requests = float("inf")
            selected_worker = 0

            for i, stats in enumerate(self.worker_stats):
                if stats["connection_status"] == "connected":
                    if stats["requests_sent"] < min_requests:
                        min_requests = stats["requests_sent"]
                        selected_worker = i

            return selected_worker

    def _get_worker_random(self) -> int:
        """Get worker randomly."""
        with self.lock:
            available_workers = [
                i
                for i, stats in enumerate(self.worker_stats)
                if stats["connection_status"] == "connected"
            ]
            if not available_workers:
                return 0  # Fallback to first worker
            return random.choice(available_workers)

    def get_worker(self) -> int:
        """Get the next worker based on the configured load balancing strategy."""
        if self.load_balance_method == "round_robin":
            return self._get_worker_round_robin()
        elif self.load_balance_method == "least_connections":
            return self._get_worker_least_connections()
        elif self.load_balance_method == "random":
            return self._get_worker_random()
        else:
            # Default to round-robin
            return self._get_worker_round_robin()

    def send_to_worker(self, worker_id: int, data) -> bool:
        """Send data to a specific detokenizer worker."""
        if (
            worker_id >= len(self.worker_sockets)
            or self.worker_sockets[worker_id] is None
        ):
            logger.error(f"❌ Invalid worker ID: {worker_id}")
            return False

        try:
            # Send the data
            self.worker_sockets[worker_id].send_pyobj(data)

            # Update statistics
            with self.lock:
                self.worker_stats[worker_id]["requests_sent"] += 1
                self.worker_stats[worker_id]["last_request_time"] = time.time()

            return True

        except Exception as e:
            logger.error(f"❌ Failed to send to worker {worker_id}: {e}")

            # Update error statistics
            with self.lock:
                self.worker_stats[worker_id]["error_count"] += 1
                if self.worker_stats[worker_id]["error_count"] > 10:
                    self.worker_stats[worker_id]["connection_status"] = "failed"
                    logger.error(
                        f"🚨 Worker {worker_id} marked as failed after 10 errors"
                    )

            return False

    def send_balanced(self, data):
        """Send data to the next available worker using load balancing."""
        max_attempts = self.num_workers
        attempts = 0

        while attempts < max_attempts:
            worker_id = self.get_worker()

            if self.send_to_worker(worker_id, data):
                return True

            attempts += 1
            logger.warning(
                f"⚠️ Failed to send to worker {worker_id}, trying next worker..."
            )

        logger.error(
            f"❌ Failed to send to any detokenizer worker after {max_attempts} attempts"
        )
        return False

    def send_pyobj(self, data):
        """Interface method expected by scheduler - maintains request affinity for streaming."""
        # Periodic cleanup to prevent memory leaks
        self.periodic_cleanup()

        # Extract request IDs from the data to maintain affinity
        request_ids = self._extract_request_ids(data)

        if not request_ids:
            # Fallback to balanced distribution if we can't extract request IDs
            logger.warning(
                f"⚠️ No request IDs found in data, using balanced distribution"
            )
            return self.send_balanced(data)

        # Check if all requests in this batch should go to the same worker
        worker_id = self._get_worker_for_requests(request_ids)

        if worker_id is not None:
            # Send all requests in batch to the same worker to maintain affinity
            success = self.send_to_worker(worker_id, data)
            if success:
                logger.debug(
                    f"🔗 Sent batch with requests {request_ids} to worker {worker_id} (affinity maintained)"
                )
                # Log worker distribution stats periodically
                if (
                    hasattr(self, "_last_stats_log")
                    and time.time() - self._last_stats_log > 30
                ):
                    self._log_worker_distribution()
                    self._last_stats_log = time.time()
            return success
        else:
            # Fallback to balanced distribution
            logger.warning(
                f"⚠️ Failed to determine worker for requests {request_ids}, using balanced distribution"
            )
            return self.send_balanced(data)

    def handle_response(self, response_data):
        """Handle response data to clean up completed requests from affinity map."""
        completed_requests = self.detect_completed_requests(response_data)
        for rid in completed_requests:
            self.cleanup_completed_request(rid)

        if completed_requests:
            logger.debug(
                f"🧹 Cleaned up {len(completed_requests)} completed requests from affinity map"
            )

    def _extract_request_ids(self, data) -> List[str]:
        """Extract request IDs from the data object."""
        try:
            if hasattr(data, "rids"):
                # BatchTokenIDOut or similar batch objects
                request_ids = data.rids
                logger.debug(
                    f"🔍 Extracted {len(request_ids)} request IDs from batch object: {request_ids[:3]}{'...' if len(request_ids) > 3 else ''}"
                )
                return request_ids
            elif hasattr(data, "rid"):
                # Single request objects
                request_ids = [data.rid]
                logger.debug(
                    f"🔍 Extracted 1 request ID from single object: {request_ids[0]}"
                )
                return request_ids
            else:
                logger.warning(
                    f"⚠️ No request ID found in data object type: {type(data)}"
                )
                return []
        except Exception as e:
            logger.error(f"❌ Error extracting request IDs from data: {e}")
            return []

    def _get_worker_for_requests(self, request_ids: List[str]) -> Optional[int]:
        """Get the worker ID that should handle these requests with proper request-level affinity."""
        if not request_ids:
            return None

        # Check if these are NEW requests (not streaming chunks)
        new_requests = []
        existing_requests = []

        for rid in request_ids:
            if rid in self.request_worker_map:
                existing_requests.append(rid)
            else:
                new_requests.append(rid)

        # Log the request analysis for debugging
        if new_requests and existing_requests:
            logger.info(
                f"🔍 Mixed request types: {len(new_requests)} new + {len(existing_requests)} existing"
            )
        elif new_requests:
            logger.info(f"🔍 All new requests: {len(new_requests)}")
        elif existing_requests:
            logger.info(f"🔍 All existing requests: {len(existing_requests)}")

        if existing_requests:
            # Use existing worker for streaming consistency
            worker_id = self.request_worker_map[existing_requests[0]]
            logger.debug(
                f"🔗 Maintaining affinity: existing requests {existing_requests} → worker {worker_id}"
            )

            # Also assign any new requests in this batch to the same worker for consistency
            if new_requests:
                for rid in new_requests:
                    self.request_worker_map[rid] = worker_id
                logger.info(
                    f"🔗 Mixed batch: new requests {new_requests} also assigned to worker {worker_id} for consistency"
                )

            return worker_id

        # NEW requests - distribute across workers using load balancing
        worker_id = self.get_worker()
        for rid in new_requests:
            self.request_worker_map[rid] = worker_id

        logger.info(
            f"🔗 New requests {new_requests} assigned to worker {worker_id} (load balanced)"
        )
        return worker_id

    def get_stats(self) -> Dict:
        """Get current load balancer statistics."""
        with self.lock:
            total_requests = sum(stats["requests_sent"] for stats in self.worker_stats)
            active_workers = sum(
                1
                for stats in self.worker_stats
                if stats["connection_status"] == "connected"
            )

            # Get affinity statistics
            affinity_stats = self.get_affinity_stats()

            return {
                "total_requests": total_requests,
                "active_workers": active_workers,
                "total_workers": self.num_workers,
                "load_balance_method": self.load_balance_method,
                "worker_details": self.worker_stats.copy(),
                "affinity_stats": affinity_stats,
            }

    def _log_worker_distribution(self):
        """Log current worker distribution for monitoring."""
        with self.lock:
            worker_distribution = self._get_worker_distribution()
            total_requests = sum(stats["requests_sent"] for stats in self.worker_stats)
            affinity_map_size = len(self.request_worker_map)

            logger.info(
                f"📊 Load Balancer Stats - Total Requests: {total_requests}, Affinity Map: {affinity_map_size}"
            )
            for worker_id, stats in enumerate(self.worker_stats):
                if stats["connection_status"] == "connected":
                    assigned_requests = worker_distribution.get(worker_id, 0)
                    logger.info(
                        f"   Worker {worker_id}: Requests Sent: {stats['requests_sent']}, Assigned: {assigned_requests}, Status: {stats['connection_status']}"
                    )
                else:
                    logger.warning(
                        f"   Worker {worker_id}: Status: {stats['connection_status']} (not receiving requests)"
                    )

    def health_check(self) -> bool:
        """Check health of all workers."""
        healthy_workers = 0

        for i, stats in enumerate(self.worker_stats):
            if stats["connection_status"] == "connected":
                healthy_workers += 1

        health_ratio = healthy_workers / self.num_workers

        if health_ratio < 0.5:
            logger.error(
                f"🚨 Critical: Only {healthy_workers}/{self.num_workers} detokenizer workers are healthy"
            )
            return False
        elif health_ratio < 1.0:
            logger.warning(
                f"⚠️ Warning: {healthy_workers}/{self.num_workers} detokenizer workers are healthy"
            )

        return True

    def cleanup_completed_request(self, request_id: str):
        """Remove completed request from affinity map to prevent memory leaks."""
        with self.lock:
            if request_id in self.request_worker_map:
                del self.request_worker_map[request_id]
                logger.debug(
                    f"🧹 Cleaned up completed request {request_id} from affinity map"
                )

    def get_affinity_stats(self) -> Dict:
        """Get statistics about request affinity."""
        with self.lock:
            return {
                "total_affinity_mappings": len(self.request_worker_map),
                "affinity_map_size": len(self.request_worker_map),
                "sample_request_ids": list(self.request_worker_map.keys())[
                    :10
                ],  # First 10 for debugging
                "worker_distribution": self._get_worker_distribution(),
            }

    def _get_worker_distribution(self) -> Dict[int, int]:
        """Get distribution of requests across workers."""
        distribution = {}
        for rid, worker_id in self.request_worker_map.items():
            distribution[worker_id] = distribution.get(worker_id, 0) + 1
        return distribution

    def detect_completed_requests(self, data) -> List[str]:
        """Detect completed requests from response data and return their IDs for cleanup."""
        completed_requests = []
        try:
            if hasattr(data, "rids") and hasattr(data, "finished_reasons"):
                # Check for finished requests
                for i, rid in enumerate(data.rids):
                    if (
                        i < len(data.finished_reasons)
                        and data.finished_reasons[i] is not None
                    ):
                        completed_requests.append(rid)
            elif hasattr(data, "rid") and hasattr(data, "finished_reason"):
                # Single request
                if data.finished_reason is not None:
                    completed_requests.append(data.rid)
        except Exception as e:
            logger.debug(f"Could not detect completed requests: {e}")

        return completed_requests

    def periodic_cleanup(self):
        """Periodically clean up old affinity mappings to prevent memory leaks."""
        current_time = time.time()
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            with self.lock:
                initial_size = len(self.request_worker_map)
                # For now, we'll keep all mappings as we don't have a way to determine
                # if a request is truly "stale" without additional context
                # This could be enhanced with request timestamps in the future
                self.last_cleanup_time = current_time
                logger.debug(
                    f"🧹 Periodic cleanup completed, affinity map size: {len(self.request_worker_map)}"
                )

    def close(self):
        """Close all worker connections."""
        for i, socket in enumerate(self.worker_sockets):
            if socket is not None:
                try:
                    socket.close()
                    logger.info(f"🔌 Closed connection to detokenizer worker {i}")
                except Exception as e:
                    logger.error(f"❌ Error closing connection to worker {i}: {e}")

        self.context.term()
        logger.info("🔌 Detokenizer load balancer closed")
