# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of this file except in compliance with the License.
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
"""Enhanced TokenizerManager for hybrid architecture with shared memory coordination."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from sglang.srt.managers.port_args import PortArgs
from sglang.srt.managers.shared_memory_manager import SharedMemoryManager
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class EnhancedTokenizerManager(TokenizerManager):
    """Enhanced TokenizerManager that coordinates with shared memory and enhanced DetokenizerManagers."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        shared_memory_manager: Optional[SharedMemoryManager] = None,
        worker_id: int = 0,
    ):
        super().__init__(server_args, port_args)

        self.worker_id = worker_id
        self.shared_memory_manager = shared_memory_manager
        self.enable_hybrid_mode = server_args.enable_multi_tokenizer

        # Request coordination
        self.active_requests = {}
        self.request_coordination = {}

        # Detokenizer assignment tracking
        self.detokenizer_assignments = {}

        logger.info(
            f"EnhancedTokenizerManager {worker_id} initialized with hybrid mode: {self.enable_hybrid_mode}"
        )

    async def _wait_one_response(self, obj, state, request):
        """Enhanced response waiting with shared memory coordination."""
        if not self.enable_hybrid_mode or not self.shared_memory_manager:
            # Fall back to original implementation
            return await super()._wait_one_response(obj, state, request)

        try:
            request_id = state.rid
            logger.debug(
                f"EnhancedTokenizerManager {self.worker_id} waiting for response for request {request_id}"
            )

            # Register request in shared memory if not already done
            if not self._is_request_registered(request_id):
                self._register_request_in_shared_memory(request_id, state, request)

            # Wait for response chunks from shared memory
            async for response_chunk in self._stream_response_from_shared_memory(
                request_id
            ):
                yield response_chunk

            # Mark request as complete in shared memory
            self.shared_memory_manager.mark_request_complete(
                request_id,
                {
                    "status": "completed",
                    "completed_by": f"EnhancedTokenizerManager_{self.worker_id}",
                },
            )

            logger.debug(
                f"EnhancedTokenizerManager {self.worker_id} completed request {request_id}"
            )

        except Exception as e:
            logger.error(
                f"Error in enhanced response waiting for request {state.rid}: {e}"
            )
            # Fall back to original implementation
            async for response in super()._wait_one_response(obj, state, request):
                yield response

    def _is_request_registered(self, request_id: str) -> bool:
        """Check if a request is already registered in shared memory."""
        if not self.shared_memory_manager:
            return False

        try:
            return self.shared_memory_manager.get_request_state(request_id) is not None
        except Exception as e:
            logger.error(f"Error checking request registration for {request_id}: {e}")
            return False

    def _register_request_in_shared_memory(self, request_id: str, state, request):
        """Register a request in shared memory."""
        if not self.shared_memory_manager:
            return

        try:
            # Extract request information
            request_info = {
                "rid": request_id,
                "status": "pending",
                "created_at": time.time(),
                "worker_id": self.worker_id,
                "request_data": {
                    "text": getattr(request, "text", ""),
                    "sampling_params": getattr(request, "sampling_params", {}),
                    "stream": getattr(request, "stream", True),
                    "model": getattr(request, "model", "unknown"),
                },
                "state_info": {
                    "prompt_tokens": getattr(state, "prompt_tokens", 0),
                    "max_tokens": getattr(state, "max_tokens", 0),
                    "current_tokens": getattr(state, "current_tokens", 0),
                },
            }

            # Register in shared memory
            success = self.shared_memory_manager.register_request(
                request_id, request_info
            )
            if success:
                logger.debug(
                    f"Request {request_id} registered in shared memory by EnhancedTokenizerManager {self.worker_id}"
                )

                # Track locally
                self.active_requests[request_id] = {
                    "state": state,
                    "request": request,
                    "registered_at": time.time(),
                }
            else:
                logger.warning(
                    f"Failed to register request {request_id} in shared memory"
                )

        except Exception as e:
            logger.error(
                f"Error registering request {request_id} in shared memory: {e}"
            )

    async def _stream_response_from_shared_memory(
        self, request_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response chunks from shared memory."""
        if not self.shared_memory_manager:
            return

        try:
            # Wait for response chunks
            while True:
                # Check if request is complete
                request_state = self.shared_memory_manager.get_request_state(request_id)
                if request_state and request_state.get("status") == "completed":
                    # Request is complete, get final response
                    final_response = request_state.get("final_response", {})
                    if final_response:
                        yield final_response
                    break

                # Get next response chunk
                response_chunk = self.shared_memory_manager.get_response_chunk(
                    request_id, timeout=0.1
                )
                if response_chunk:
                    yield response_chunk

                    # If this is the final response, break
                    if response_chunk.get("type") == "final_response":
                        break

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(
                f"Error streaming response from shared memory for request {request_id}: {e}"
            )

    def assign_detokenizer(self, request_id: str, detokenizer_id: int) -> bool:
        """Assign a DetokenizerManager to handle a specific request."""
        try:
            self.detokenizer_assignments[request_id] = {
                "detokenizer_id": detokenizer_id,
                "assigned_at": time.time(),
                "worker_id": self.worker_id,
            }

            logger.debug(
                f"Request {request_id} assigned to DetokenizerManager {detokenizer_id} by EnhancedTokenizerManager {self.worker_id}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error assigning DetokenizerManager for request {request_id}: {e}"
            )
            return False

    def get_detokenizer_assignment(self, request_id: str) -> Optional[int]:
        """Get the DetokenizerManager assignment for a request."""
        assignment = self.detokenizer_assignments.get(request_id)
        return assignment["detokenizer_id"] if assignment else None

    def update_request_coordination(
        self, request_id: str, coordination_data: Dict[str, Any]
    ) -> bool:
        """Update coordination information for a request."""
        try:
            if request_id not in self.request_coordination:
                self.request_coordination[request_id] = {}

            self.request_coordination[request_id].update(coordination_data)
            self.request_coordination[request_id]["last_updated"] = time.time()

            logger.debug(
                f"Updated coordination for request {request_id}: {coordination_data}"
            )
            return True

        except Exception as e:
            logger.error(f"Error updating coordination for request {request_id}: {e}")
            return False

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get statistics for this worker."""
        return {
            "worker_id": self.worker_id,
            "active_requests": len(self.active_requests),
            "detokenizer_assignments": len(self.detokenizer_assignments),
            "request_coordination": len(self.request_coordination),
            "enable_hybrid_mode": self.enable_hybrid_mode,
            "shared_memory_available": self.shared_memory_manager is not None,
        }

    def cleanup_worker(self):
        """Clean up worker resources."""
        try:
            # Clean up local state
            self.active_requests.clear()
            self.detokenizer_assignments.clear()
            self.request_coordination.clear()

            logger.info(f"EnhancedTokenizerManager {self.worker_id} cleanup complete")

        except Exception as e:
            logger.error(
                f"Error cleaning up EnhancedTokenizerManager {self.worker_id}: {e}"
            )

    def handle_request_completion(
        self, request_id: str, completion_data: Dict[str, Any]
    ) -> bool:
        """Handle request completion notification from DetokenizerManager."""
        try:
            if request_id in self.active_requests:
                # Update local state
                self.active_requests[request_id]["completion_data"] = completion_data
                self.active_requests[request_id]["completed_at"] = time.time()

                # Update shared memory
                if self.shared_memory_manager:
                    self.shared_memory_manager.update_request_state(
                        request_id,
                        {
                            "status": "completed",
                            "completion_data": completion_data,
                            "completed_at": time.time(),
                        },
                    )

                logger.debug(
                    f"Request {request_id} completion handled by EnhancedTokenizerManager {self.worker_id}"
                )
                return True
            else:
                logger.warning(
                    f"Request {request_id} not found in active requests for EnhancedTokenizerManager {self.worker_id}"
                )
                return False

        except Exception as e:
            logger.error(f"Error handling request completion for {request_id}: {e}")
            return False
