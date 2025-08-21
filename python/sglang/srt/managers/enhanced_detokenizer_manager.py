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
"""Enhanced DetokenizerManager for hybrid architecture with HTTP response formatting."""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.port_args import PortArgs
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.managers.shared_memory_manager import SharedMemoryManager
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class EnhancedDetokenizerManager(DetokenizerManager):
    """Enhanced DetokenizerManager that can handle HTTP response formatting."""

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
        self.enable_http_formatting = server_args.enable_multi_tokenizer

        # Request affinity tracking
        self.request_affinity = {}

        # Response formatting cache
        self.response_cache = {}

        logger.info(
            f"EnhancedDetokenizerManager {worker_id} initialized with HTTP formatting: {self.enable_http_formatting}"
        )

    def handle_batch_token_id_out(self, recv_obj):
        """Enhanced handler for batch token output with HTTP formatting capability."""
        # First, call the parent method to handle the basic detokenization
        result = super().handle_batch_token_id_out(recv_obj)

        # If HTTP formatting is enabled, format the response
        if self.enable_http_formatting and self.shared_memory_manager:
            self._handle_http_response_formatting(recv_obj, result)

        return result

    def _handle_http_response_formatting(self, recv_obj, detokenized_result):
        """Handle HTTP response formatting for streaming responses."""
        try:
            for rid in recv_obj.rids:
                if rid not in self.request_affinity:
                    # This request is not assigned to this DM, skip
                    continue

                # Get the request state from shared memory
                request_state = self.shared_memory_manager.get_request_state(rid)
                if not request_state:
                    logger.warning(f"Request {rid} not found in shared memory")
                    continue

                # Format the response based on the request type
                formatted_response = self._format_http_response(
                    rid, detokenized_result, request_state
                )

                # Add the formatted response to the shared memory
                if formatted_response:
                    self.shared_memory_manager.add_response_chunk(
                        rid, formatted_response
                    )

                    # Check if this is the final response
                    if self._is_response_complete(
                        rid, detokenized_result, request_state
                    ):
                        final_response = self._create_final_response(rid, request_state)
                        self.shared_memory_manager.mark_request_complete(
                            rid, final_response
                        )

                        # Clean up local state
                        if rid in self.request_affinity:
                            del self.request_affinity[rid]
                        if rid in self.response_cache:
                            del self.response_cache[rid]

        except Exception as e:
            logger.error(f"Error in HTTP response formatting: {e}")

    def _format_http_response(
        self, rid: str, detokenized_result, request_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Format the detokenized result as an HTTP response chunk."""
        try:
            # Extract the text for this specific request
            text_chunk = self._extract_text_for_request(rid, detokenized_result)
            if not text_chunk:
                return None

            # Get the original request parameters
            sampling_params = request_state.get("sampling_params", {})
            stream = request_state.get("stream", True)

            # Format the response based on the request type
            if stream:
                # Streaming response
                response = {
                    "type": "stream_chunk",
                    "id": str(uuid.uuid4()),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request_state.get("model", "unknown"),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text_chunk},
                            "finish_reason": None,
                        }
                    ],
                }
            else:
                # Non-streaming response
                response = {
                    "type": "complete_response",
                    "id": str(uuid.uuid4()),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request_state.get("model", "unknown"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text_chunk},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": request_state.get("prompt_tokens", 0),
                        "completion_tokens": request_state.get("completion_tokens", 0),
                        "total_tokens": request_state.get("total_tokens", 0),
                    },
                }

            # Cache the response for potential retransmission
            if rid not in self.response_cache:
                self.response_cache[rid] = []
            self.response_cache[rid].append(response)

            return response

        except Exception as e:
            logger.error(f"Error formatting HTTP response for request {rid}: {e}")
            return None

    def _extract_text_for_request(self, rid: str, detokenized_result) -> Optional[str]:
        """Extract the text chunk for a specific request from the detokenized result."""
        try:
            # This is a simplified version - in practice, you'd need to match
            # the specific text output for this request ID
            if hasattr(detokenized_result, "texts") and rid in detokenized_result.texts:
                return detokenized_result.texts[rid]
            elif hasattr(detokenized_result, "text"):
                return detokenized_result.text
            else:
                # Fallback: try to extract from the result object
                return str(detokenized_result)
        except Exception as e:
            logger.error(f"Error extracting text for request {rid}: {e}")
            return None

    def _is_response_complete(
        self, rid: str, detokenized_result, request_state: Dict[str, Any]
    ) -> bool:
        """Check if the response for a request is complete."""
        try:
            # Check if the detokenized result indicates completion
            if (
                hasattr(detokenized_result, "finished_reasons")
                and rid in detokenized_result.finished_reasons
            ):
                return detokenized_result.finished_reasons[rid] is not None
            elif hasattr(detokenized_result, "finished_reason"):
                return detokenized_result.finished_reason is not None
            else:
                # Fallback: check request state
                return request_state.get("status") == "completed"
        except Exception as e:
            logger.error(f"Error checking response completion for request {rid}: {e}")
            return False

    def _create_final_response(
        self, rid: str, request_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create the final response for a completed request."""
        try:
            # Get all cached responses for this request
            cached_responses = self.response_cache.get(rid, [])

            # Combine all text chunks
            full_text = ""
            for response in cached_responses:
                if response.get("type") == "stream_chunk":
                    content = (
                        response.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    full_text += content

            # Create final response
            final_response = {
                "type": "final_response",
                "id": str(uuid.uuid4()),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request_state.get("model", "unknown"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": full_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": request_state.get("prompt_tokens", 0),
                    "completion_tokens": request_state.get("completion_tokens", 0),
                    "total_tokens": request_state.get("total_tokens", 0),
                },
            }

            return final_response

        except Exception as e:
            logger.error(f"Error creating final response for request {rid}: {e}")
            return {"error": f"Failed to create final response: {e}"}

    def assign_request(self, rid: str) -> bool:
        """Assign a request to this DetokenizerManager for affinity."""
        try:
            self.request_affinity[rid] = {
                "assigned_at": time.time(),
                "worker_id": self.worker_id,
            }
            logger.debug(
                f"Request {rid} assigned to EnhancedDetokenizerManager {self.worker_id}"
            )
            return True
        except Exception as e:
            logger.error(f"Error assigning request {rid}: {e}")
            return False

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get statistics for this worker."""
        return {
            "worker_id": self.worker_id,
            "assigned_requests": len(self.request_affinity),
            "cached_responses": len(self.response_cache),
            "enable_http_formatting": self.enable_http_formatting,
        }

    def cleanup_worker(self):
        """Clean up worker resources."""
        try:
            # Clean up local state
            self.request_affinity.clear()
            self.response_cache.clear()

            logger.info(f"EnhancedDetokenizerManager {self.worker_id} cleanup complete")
        except Exception as e:
            logger.error(
                f"Error cleaning up EnhancedDetokenizerManager {self.worker_id}: {e}"
            )
