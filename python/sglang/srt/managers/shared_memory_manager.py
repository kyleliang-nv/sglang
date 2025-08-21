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
"""Shared memory manager for multi-process TokenizerManager."""

import logging
import multiprocessing as mp
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SharedMemoryManager:
    """Manages shared memory for multi-process TokenizerManager architecture."""

    def __init__(self, max_requests: int = 1000):
        self.max_requests = max_requests

        # Shared memory for request states
        self.request_states = mp.Manager().dict()

        # Shared memory for response queues
        self.response_queues = mp.Manager().dict()

        # Shared memory for request tracking
        self.request_tracking = mp.Manager().dict()

        # Lock for thread-safe operations
        self._lock = threading.Lock()

        # Cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_old_requests, daemon=True
        )
        self._cleanup_thread.start()

        logger.info(f"SharedMemoryManager initialized with max_requests={max_requests}")

    def register_request(self, request_id: str, initial_state: Dict[str, Any]) -> bool:
        """Register a new request in shared memory."""
        with self._lock:
            if len(self.request_states) >= self.max_requests:
                logger.warning(
                    f"Max requests reached ({self.max_requests}), cannot register {request_id}"
                )
                return False

            self.request_states[request_id] = initial_state
            self.request_tracking[request_id] = {
                "created_at": time.time(),
                "last_updated": time.time(),
                "status": "pending",
            }

            # Initialize response queue for this request
            self.response_queues[request_id] = mp.Queue()

            logger.debug(f"Registered request {request_id} in shared memory")
            return True

    def update_request_state(self, request_id: str, updates: Dict[str, Any]) -> bool:
        """Update request state in shared memory."""
        with self._lock:
            if request_id not in self.request_states:
                logger.warning(f"Request {request_id} not found in shared memory")
                return False

            # Update the state
            self.request_states[request_id].update(updates)

            # Update tracking info
            if request_id in self.request_tracking:
                self.request_tracking[request_id]["last_updated"] = time.time()
                if "status" in updates:
                    self.request_tracking[request_id]["status"] = updates["status"]

            logger.debug(f"Updated request {request_id} state: {updates}")
            return True

    def get_request_state(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get request state from shared memory."""
        with self._lock:
            return self.request_states.get(request_id)

    def add_response_chunk(self, request_id: str, chunk: Dict[str, Any]) -> bool:
        """Add a response chunk to the request's response queue."""
        with self._lock:
            if request_id not in self.response_queues:
                logger.warning(f"Request {request_id} not found in response queues")
                return False

            try:
                self.response_queues[request_id].put(chunk, block=False)
                logger.debug(f"Added response chunk for request {request_id}")
                return True
            except Exception as e:
                logger.error(
                    f"Failed to add response chunk for request {request_id}: {e}"
                )
                return False

    def get_response_chunk(
        self, request_id: str, timeout: float = 0.1
    ) -> Optional[Dict[str, Any]]:
        """Get a response chunk from the request's response queue."""
        with self._lock:
            if request_id not in self.response_queues:
                return None

            try:
                return self.response_queues[request_id].get(timeout=timeout)
            except:
                return None

    def mark_request_complete(
        self, request_id: str, final_response: Dict[str, Any]
    ) -> bool:
        """Mark a request as complete and add final response."""
        with self._lock:
            if request_id not in self.request_states:
                return False

            # Update state to completed
            self.update_request_state(
                request_id,
                {
                    "status": "completed",
                    "final_response": final_response,
                    "completed_at": time.time(),
                },
            )

            # Add final response to queue
            self.add_response_chunk(
                request_id,
                {
                    "type": "final_response",
                    "data": final_response,
                    "timestamp": time.time(),
                },
            )

            logger.info(f"Marked request {request_id} as complete")
            return True

    def cleanup_request(self, request_id: str) -> bool:
        """Clean up a completed request from shared memory."""
        with self._lock:
            if request_id in self.request_states:
                del self.request_states[request_id]

            if request_id in self.response_queues:
                del self.response_queues[request_id]

            if request_id in self.request_tracking:
                del self.request_tracking[request_id]

            logger.debug(f"Cleaned up request {request_id}")
            return True

    def get_request_stats(self) -> Dict[str, Any]:
        """Get statistics about current requests."""
        with self._lock:
            total_requests = len(self.request_states)
            pending_requests = sum(
                1
                for tracking in self.request_tracking.values()
                if tracking.get("status") == "pending"
            )
            running_requests = sum(
                1
                for tracking in self.request_tracking.values()
                if tracking.get("status") == "running"
            )
            completed_requests = sum(
                1
                for tracking in self.request_tracking.values()
                if tracking.get("status") == "completed"
            )

            return {
                "total_requests": total_requests,
                "pending_requests": pending_requests,
                "running_requests": running_requests,
                "completed_requests": completed_requests,
                "max_requests": self.max_requests,
            }

    def _cleanup_old_requests(self):
        """Background thread to cleanup old completed requests."""
        while True:
            try:
                time.sleep(60)  # Check every minute

                current_time = time.time()
                to_cleanup = []

                with self._lock:
                    for request_id, tracking in self.request_tracking.items():
                        # Clean up requests that have been completed for more than 5 minutes
                        if (
                            tracking.get("status") == "completed"
                            and current_time - tracking.get("completed_at", 0) > 300
                        ):
                            to_cleanup.append(request_id)

                # Clean up outside the lock
                for request_id in to_cleanup:
                    self.cleanup_request(request_id)

                if to_cleanup:
                    logger.debug(f"Cleaned up {len(to_cleanup)} old requests")

            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")

    def shutdown(self):
        """Shutdown the shared memory manager."""
        logger.info("Shutting down SharedMemoryManager")

        # Clean up all requests
        with self._lock:
            request_ids = list(self.request_states.keys())

        for request_id in request_ids:
            self.cleanup_request(request_id)

        logger.info("SharedMemoryManager shutdown complete")
