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

import asyncio
import logging
import multiprocessing
import time
from typing import Any, Dict, Optional

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.shared_memory_manager import SharedMemoryManager
from sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger("sglang.srt.managers.enhanced_detokenizer_manager")


class EnhancedDetokenizerManager(DetokenizerManager):
    """
    Enhanced DetokenizerManager that can handle HTTP response formatting
    and work with the hybrid architecture.
    """

    def __init__(
        self, server_args: ServerArgs, port_args: PortArgs, worker_id: int = 0
    ):
        super().__init__(server_args, port_args)
        self.worker_id = worker_id
        self.shared_memory = SharedMemoryManager()
        self.is_hybrid = getattr(server_args, "enable_multi_tokenizer", False)

        if self.is_hybrid:
            logger.info(
                f"🚀 EnhancedDetokenizerManager worker {worker_id} initialized in hybrid mode"
            )
        else:
            logger.info(
                f"🚀 EnhancedDetokenizerManager worker {worker_id} initialized in standard mode"
            )

    def process_response(
        self, request_id: str, response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process and format the response for HTTP output.
        This is the key enhancement that offloads work from TokenizerManager.
        """
        try:
            # Format the response for HTTP output
            formatted_response = {
                "request_id": request_id,
                "status": "success",
                "data": response_data,
                "timestamp": time.time(),
                "worker_id": self.worker_id,
            }

            # Store in shared memory for TokenizerManager to retrieve
            self.shared_memory.store_response(request_id, formatted_response)

            logger.debug(
                f"📤 EnhancedDetokenizerManager worker {self.worker_id} formatted response for {request_id}"
            )
            return formatted_response

        except Exception as e:
            logger.error(
                f"❌ EnhancedDetokenizerManager worker {self.worker_id} failed to process response: {e}"
            )
            error_response = {
                "request_id": request_id,
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
                "worker_id": self.worker_id,
            }
            self.shared_memory.store_response(request_id, error_response)
            return error_response

    def handle_completion(self, request_id: str, completion_data: Dict[str, Any]):
        """
        Handle completion notifications and format responses.
        """
        if self.is_hybrid:
            # Process the completion and format the response
            formatted_response = self.process_response(request_id, completion_data)
            logger.info(
                f"✅ EnhancedDetokenizerManager worker {self.worker_id} completed request {request_id}"
            )
        else:
            # Fall back to standard behavior
            super().handle_completion(request_id, completion_data)

    def run(self):
        """
        Enhanced run method that handles hybrid mode.
        """
        if self.is_hybrid:
            logger.info(
                f"🚀 EnhancedDetokenizerManager worker {self.worker_id} starting in hybrid mode"
            )
            try:
                # Initialize shared memory
                self.shared_memory.initialize()
                logger.info(
                    f"✅ EnhancedDetokenizerManager worker {self.worker_id} shared memory initialized"
                )

                # Run the main event loop
                self.event_loop()

            except Exception as e:
                logger.error(
                    f"❌ EnhancedDetokenizerManager worker {self.worker_id} failed: {e}"
                )
                raise
        else:
            # Standard mode
            logger.info(
                f"🚀 EnhancedDetokenizerManager worker {self.worker_id} starting in standard mode"
            )
            super().run()
