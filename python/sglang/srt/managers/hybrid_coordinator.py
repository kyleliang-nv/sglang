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
"""Hybrid coordinator for managing enhanced TokenizerManagers and DetokenizerManagers."""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.managers.enhanced_detokenizer_manager import EnhancedDetokenizerManager
from sglang.srt.managers.enhanced_tokenizer_manager import EnhancedTokenizerManager
from sglang.srt.managers.port_args import PortArgs
from sglang.srt.managers.shared_memory_manager import SharedMemoryManager
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class HybridCoordinator:
    """Coordinates the hybrid architecture between enhanced TokenizerManagers and DetokenizerManagers."""

    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        self.enable_hybrid = server_args.enable_multi_tokenizer

        # Shared memory manager
        self.shared_memory_manager = None
        if self.enable_hybrid:
            self.shared_memory_manager = SharedMemoryManager(
                max_requests=server_args.max_queued_requests or 1000
            )

        # Worker management
        self.tokenizer_workers: Dict[int, EnhancedTokenizerManager] = {}
        self.detokenizer_workers: Dict[int, EnhancedDetokenizerManager] = {}

        # Request routing
        self.request_routing: Dict[str, Dict[str, Any]] = {}

        # Load balancing
        self.load_balance_policy = server_args.detokenizer_load_balance_policy
        self.detokenizer_loads: Dict[int, int] = {}

        # Coordination state
        self.coordination_active = False
        self.coordination_task = None

        logger.info(
            f"HybridCoordinator initialized with hybrid mode: {self.enable_hybrid}"
        )

    def add_tokenizer_worker(
        self, worker_id: int, tokenizer_manager: EnhancedTokenizerManager
    ) -> bool:
        """Add a TokenizerManager worker to the coordinator."""
        try:
            if not self.enable_hybrid:
                logger.warning(
                    "Hybrid mode not enabled, cannot add TokenizerManager worker"
                )
                return False

            # Set the shared memory manager
            tokenizer_manager.shared_memory_manager = self.shared_memory_manager

            # Register the worker
            self.tokenizer_workers[worker_id] = tokenizer_manager

            logger.info(
                f"TokenizerManager worker {worker_id} added to HybridCoordinator"
            )
            return True

        except Exception as e:
            logger.error(f"Error adding TokenizerManager worker {worker_id}: {e}")
            return False

    def add_detokenizer_worker(
        self, worker_id: int, detokenizer_manager: EnhancedDetokenizerManager
    ) -> bool:
        """Add a DetokenizerManager worker to the coordinator."""
        try:
            if not self.enable_hybrid:
                logger.warning(
                    "Hybrid mode not enabled, cannot add DetokenizerManager worker"
                )
                return False

            # Set the shared memory manager
            detokenizer_manager.shared_memory_manager = self.shared_memory_manager

            # Register the worker
            self.detokenizer_workers[worker_id] = detokenizer_manager

            # Initialize load tracking
            self.detokenizer_loads[worker_id] = 0

            logger.info(
                f"DetokenizerManager worker {worker_id} added to HybridCoordinator"
            )
            return True

        except Exception as e:
            logger.error(f"Error adding DetokenizerManager worker {worker_id}: {e}")
            return False

    def route_request(
        self, request_id: str, request_data: Dict[str, Any]
    ) -> Tuple[Optional[int], Optional[int]]:
        """Route a request to appropriate TokenizerManager and DetokenizerManager workers."""
        if not self.enable_hybrid:
            return None, None

        try:
            # Select TokenizerManager worker (round-robin for now)
            tokenizer_worker_id = self._select_tokenizer_worker(request_id)

            # Select DetokenizerManager worker based on load balancing policy
            detokenizer_worker_id = self._select_detokenizer_worker(
                request_id, request_data
            )

            if tokenizer_worker_id is not None and detokenizer_worker_id is not None:
                # Record the routing
                self.request_routing[request_id] = {
                    "tokenizer_worker_id": tokenizer_worker_id,
                    "detokenizer_worker_id": detokenizer_worker_id,
                    "routed_at": time.time(),
                    "request_data": request_data,
                }

                # Assign request to DetokenizerManager for affinity
                if detokenizer_worker_id in self.detokenizer_workers:
                    self.detokenizer_workers[detokenizer_worker_id].assign_request(
                        request_id
                    )

                # Update load tracking
                if detokenizer_worker_id in self.detokenizer_loads:
                    self.detokenizer_loads[detokenizer_worker_id] += 1

                logger.debug(
                    f"Request {request_id} routed to TM worker {tokenizer_worker_id} and DM worker {detokenizer_worker_id}"
                )

                return tokenizer_worker_id, detokenizer_worker_id
            else:
                logger.warning(
                    f"Could not route request {request_id}: TM worker {tokenizer_worker_id}, DM worker {detokenizer_worker_id}"
                )
                return None, None

        except Exception as e:
            logger.error(f"Error routing request {request_id}: {e}")
            return None, None

    def _select_tokenizer_worker(self, request_id: str) -> Optional[int]:
        """Select a TokenizerManager worker for the request."""
        if not self.tokenizer_workers:
            return None

        # Simple round-robin selection
        worker_ids = list(self.tokenizer_workers.keys())
        if not worker_ids:
            return None

        # Use request_id hash for consistent assignment
        hash_value = hash(request_id)
        selected_worker = worker_ids[hash_value % len(worker_ids)]

        return selected_worker

    def _select_detokenizer_worker(
        self, request_id: str, request_data: Dict[str, Any]
    ) -> Optional[int]:
        """Select a DetokenizerManager worker based on load balancing policy."""
        if not self.detokenizer_workers:
            return None

        worker_ids = list(self.detokenizer_workers.keys())
        if not worker_ids:
            return None

        if self.load_balance_policy == "round_robin":
            # Round-robin selection
            hash_value = hash(request_id)
            selected_worker = worker_ids[hash_value % len(worker_ids)]

        elif self.load_balance_policy == "least_loaded":
            # Select worker with lowest load
            selected_worker = min(
                worker_ids, key=lambda wid: self.detokenizer_loads.get(wid, 0)
            )

        elif self.load_balance_policy == "weighted":
            # Weighted selection based on load and other factors
            selected_worker = self._weighted_selection(worker_ids, request_data)

        else:
            # Default to round-robin
            hash_value = hash(request_id)
            selected_worker = worker_ids[hash_value % len(worker_ids)]

        return selected_worker

    def _weighted_selection(
        self, worker_ids: List[int], request_data: Dict[str, Any]
    ) -> int:
        """Weighted selection of DetokenizerManager worker."""
        if not worker_ids:
            return 0

        # Calculate weights based on load and request characteristics
        weights = []
        for worker_id in worker_ids:
            base_weight = 1.0 / (self.detokenizer_loads.get(worker_id, 0) + 1)

            # Additional weight factors could be added here
            # For example, based on request size, priority, etc.

            weights.append(base_weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(worker_ids)] * len(worker_ids)

        # Select based on weights
        import random

        selected_worker = random.choices(worker_ids, weights=weights)[0]

        return selected_worker

    def mark_request_complete(self, request_id: str, worker_id: int) -> bool:
        """Mark a request as complete and update load tracking."""
        try:
            if request_id in self.request_routing:
                # Get the DetokenizerManager worker that handled this request
                dm_worker_id = self.request_routing[request_id].get(
                    "detokenizer_worker_id"
                )

                if dm_worker_id is not None and dm_worker_id in self.detokenizer_loads:
                    # Decrease load
                    self.detokenizer_loads[dm_worker_id] = max(
                        0, self.detokenizer_loads[dm_worker_id] - 1
                    )

                # Remove routing entry
                del self.request_routing[request_id]

                logger.debug(
                    f"Request {request_id} marked complete, updated load for DM worker {dm_worker_id}"
                )
                return True
            else:
                logger.warning(f"Request {request_id} not found in routing table")
                return False

        except Exception as e:
            logger.error(f"Error marking request {request_id} complete: {e}")
            return False

    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics."""
        return {
            "enable_hybrid": self.enable_hybrid,
            "tokenizer_workers": len(self.tokenizer_workers),
            "detokenizer_workers": len(self.detokenizer_workers),
            "active_requests": len(self.request_routing),
            "detokenizer_loads": dict(self.detokenizer_loads),
            "load_balance_policy": self.load_balance_policy,
            "shared_memory_available": self.shared_memory_manager is not None,
            "coordination_active": self.coordination_active,
        }

    def start_coordination(self):
        """Start the coordination background task."""
        if not self.enable_hybrid or self.coordination_active:
            return

        try:
            self.coordination_active = True
            self.coordination_task = asyncio.create_task(self._coordination_loop())
            logger.info("HybridCoordinator coordination started")

        except Exception as e:
            logger.error(f"Error starting coordination: {e}")
            self.coordination_active = False

    def stop_coordination(self):
        """Stop the coordination background task."""
        try:
            self.coordination_active = False
            if self.coordination_task:
                self.coordination_task.cancel()
                self.coordination_task = None

            logger.info("HybridCoordinator coordination stopped")

        except Exception as e:
            logger.error(f"Error stopping coordination: {e}")

    async def _coordination_loop(self):
        """Main coordination loop."""
        try:
            while self.coordination_active:
                # Monitor worker health
                await self._monitor_workers()

                # Clean up completed requests
                await self._cleanup_completed_requests()

                # Update load balancing information
                await self._update_load_balancing()

                # Small delay
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Coordination loop cancelled")
        except Exception as e:
            logger.error(f"Error in coordination loop: {e}")
            self.coordination_active = False

    async def _monitor_workers(self):
        """Monitor worker health and status."""
        try:
            # Check TokenizerManager workers
            for worker_id, worker in self.tokenizer_workers.items():
                if not hasattr(worker, "is_alive") or not worker.is_alive():
                    logger.warning(
                        f"TokenizerManager worker {worker_id} appears to be down"
                    )

            # Check DetokenizerManager workers
            for worker_id, worker in self.detokenizer_workers.items():
                if not hasattr(worker, "is_alive") or not worker.is_alive():
                    logger.warning(
                        f"DetokenizerManager worker {worker_id} appears to be down"
                    )

        except Exception as e:
            logger.error(f"Error monitoring workers: {e}")

    async def _cleanup_completed_requests(self):
        """Clean up completed requests from routing table."""
        try:
            current_time = time.time()
            to_remove = []

            for request_id, routing_info in self.request_routing.items():
                # Remove requests that have been routing for more than 10 minutes
                if current_time - routing_info.get("routed_at", 0) > 600:
                    to_remove.append(request_id)

            for request_id in to_remove:
                self.mark_request_complete(request_id, 0)

        except Exception as e:
            logger.error(f"Error cleaning up completed requests: {e}")

    async def _update_load_balancing(self):
        """Update load balancing information."""
        try:
            # Update load information from workers
            for worker_id, worker in self.detokenizer_workers.items():
                if hasattr(worker, "get_worker_stats"):
                    stats = worker.get_worker_stats()
                    assigned_requests = stats.get("assigned_requests", 0)
                    self.detokenizer_loads[worker_id] = assigned_requests

        except Exception as e:
            logger.error(f"Error updating load balancing: {e}")

    def shutdown(self):
        """Shutdown the hybrid coordinator."""
        try:
            logger.info("Shutting down HybridCoordinator")

            # Stop coordination
            self.stop_coordination()

            # Clean up workers
            for worker in self.tokenizer_workers.values():
                if hasattr(worker, "cleanup_worker"):
                    worker.cleanup_worker()

            for worker in self.detokenizer_workers.values():
                if hasattr(worker, "cleanup_worker"):
                    worker.cleanup_worker()

            # Clean up shared memory
            if self.shared_memory_manager:
                self.shared_memory_manager.shutdown()

            logger.info("HybridCoordinator shutdown complete")

        except Exception as e:
            logger.error(f"Error shutting down HybridCoordinator: {e}")
