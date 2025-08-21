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
"""Enhanced HTTP server with hybrid architecture support."""

import asyncio
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.enhanced_detokenizer_manager import EnhancedDetokenizerManager
from sglang.srt.managers.enhanced_tokenizer_manager import EnhancedTokenizerManager
from sglang.srt.managers.hybrid_coordinator import HybridCoordinator
from sglang.srt.managers.port_args import PortArgs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class EnhancedHTTPServer:
    """Enhanced HTTP server with hybrid architecture support."""

    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        self.enable_hybrid = server_args.enable_multi_tokenizer

        # FastAPI app
        self.app = FastAPI(title="SGLang Enhanced Server", version="1.0.0")

        # Hybrid coordinator
        self.coordinator = None
        if self.enable_hybrid:
            self.coordinator = HybridCoordinator(server_args)

        # Worker processes
        self.tokenizer_workers: Dict[int, mp.Process] = {}
        self.detokenizer_workers: Dict[int, mp.Process] = {}
        self.scheduler_processes: Dict[int, mp.Process] = {}

        # Process management
        self.processes_started = False
        self.shutdown_event = mp.Event()

        # Setup routes
        self._setup_routes()

        logger.info(
            f"EnhancedHTTPServer initialized with hybrid mode: {self.enable_hybrid}"
        )

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.post("/generate")
        async def generate(request: Request):
            """Enhanced generate endpoint with hybrid architecture support."""
            try:
                # Parse request
                request_data = await request.json()
                request_id = request_data.get(
                    "request_id", f"req_{int(time.time() * 1000)}"
                )

                if self.enable_hybrid and self.coordinator:
                    # Use hybrid architecture
                    return await self._handle_hybrid_generate(
                        request_id, request_data, request
                    )
                else:
                    # Fall back to standard implementation
                    return await self._handle_standard_generate(
                        request_id, request_data, request
                    )

            except Exception as e:
                logger.error(f"Error in generate endpoint: {e}")
                return Response(
                    content=f'{{"error": "Internal server error: {str(e)}"}}',
                    status_code=500,
                    media_type="application/json",
                )

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            if self.enable_hybrid and self.coordinator:
                stats = self.coordinator.get_coordination_stats()
                return {
                    "status": "healthy",
                    "hybrid_mode": True,
                    "coordinator_stats": stats,
                }
            else:
                return {"status": "healthy", "hybrid_mode": False}

        @self.app.get("/stats")
        async def stats():
            """Statistics endpoint."""
            if self.enable_hybrid and self.coordinator:
                return {
                    "coordinator_stats": self.coordinator.get_coordination_stats(),
                    "worker_stats": self._get_worker_stats(),
                }
            else:
                return {"message": "Hybrid mode not enabled"}

    async def _handle_hybrid_generate(
        self, request_id: str, request_data: Dict[str, Any], request: Request
    ) -> Response:
        """Handle generate request using hybrid architecture."""
        try:
            # Route request through coordinator
            tm_worker_id, dm_worker_id = self.coordinator.route_request(
                request_id, request_data
            )

            if tm_worker_id is None or dm_worker_id is None:
                return Response(
                    content='{"error": "Failed to route request"}',
                    status_code=503,
                    media_type="application/json",
                )

            # Check if streaming is requested
            stream = request_data.get("stream", True)

            if stream:
                # Streaming response
                return StreamingResponse(
                    self._stream_hybrid_response(request_id, tm_worker_id),
                    media_type="text/plain",
                )
            else:
                # Non-streaming response
                response = await self._get_hybrid_response(request_id, tm_worker_id)
                return Response(content=response, media_type="application/json")

        except Exception as e:
            logger.error(f"Error in hybrid generate: {e}")
            return Response(
                content=f'{{"error": "Hybrid generation error: {str(e)}"}}',
                status_code=500,
                media_type="application/json",
            )

    async def _handle_standard_generate(
        self, request_id: str, request_data: Dict[str, Any], request: Request
    ) -> Response:
        """Handle generate request using standard implementation."""
        # This would integrate with the existing engine
        # For now, return a placeholder response
        return Response(
            content='{"message": "Standard generation not yet implemented"}',
            media_type="application/json",
        )

    async def _stream_hybrid_response(self, request_id: str, tm_worker_id: int):
        """Stream response from hybrid architecture."""
        try:
            # Get the TokenizerManager worker
            if tm_worker_id not in self.coordinator.tokenizer_workers:
                yield f"Error: TokenizerManager worker {tm_worker_id} not found"
                return

            tm_worker = self.coordinator.tokenizer_workers[tm_worker_id]

            # Stream response chunks
            async for chunk in tm_worker._stream_response_from_shared_memory(
                request_id
            ):
                if chunk.get("type") == "stream_chunk":
                    # Format streaming chunk
                    formatted_chunk = self._format_streaming_chunk(chunk)
                    yield formatted_chunk
                elif chunk.get("type") == "final_response":
                    # Final response
                    formatted_final = self._format_final_response(chunk)
                    yield formatted_final
                    break

        except Exception as e:
            logger.error(f"Error streaming hybrid response: {e}")
            yield f"Error: {str(e)}"

    async def _get_hybrid_response(self, request_id: str, tm_worker_id: int) -> str:
        """Get non-streaming response from hybrid architecture."""
        try:
            # Get the TokenizerManager worker
            if tm_worker_id not in self.coordinator.tokenizer_workers:
                return '{"error": "TokenizerManager worker not found"}'

            tm_worker = self.coordinator.tokenizer_workers[tm_worker_id]

            # Wait for final response
            final_response = None
            async for chunk in tm_worker._stream_response_from_shared_memory(
                request_id
            ):
                if chunk.get("type") == "final_response":
                    final_response = chunk
                    break

            if final_response:
                return self._format_final_response(final_response)
            else:
                return '{"error": "No response received"}'

        except Exception as e:
            logger.error(f"Error getting hybrid response: {e}")
            return f'{{"error": "Response error: {str(e)}"}}'

    def _format_streaming_chunk(self, chunk: Dict[str, Any]) -> str:
        """Format streaming chunk for HTTP response."""
        try:
            # Convert to Server-Sent Events format
            data = chunk.get("data", chunk)
            return f"data: {data}\n\n"
        except Exception as e:
            logger.error(f"Error formatting streaming chunk: {e}")
            return f'data: {{"error": "Formatting error"}}\n\n'

    def _format_final_response(self, chunk: Dict[str, Any]) -> str:
        """Format final response for HTTP response."""
        try:
            data = chunk.get("data", chunk)
            return data
        except Exception as e:
            logger.error(f"Error formatting final response: {e}")
            return '{"error": "Formatting error"}'

    def _get_worker_stats(self) -> Dict[str, Any]:
        """Get statistics from all workers."""
        stats = {}

        try:
            # TokenizerManager worker stats
            if self.coordinator:
                for worker_id, worker in self.coordinator.tokenizer_workers.items():
                    if hasattr(worker, "get_worker_stats"):
                        stats[f"tm_worker_{worker_id}"] = worker.get_worker_stats()

                # DetokenizerManager worker stats
                for worker_id, worker in self.coordinator.detokenizer_workers.items():
                    if hasattr(worker, "get_worker_stats"):
                        stats[f"dm_worker_{worker_id}"] = worker.get_worker_stats()
        except Exception as e:
            logger.error(f"Error getting worker stats: {e}")
            stats["error"] = str(e)

        return stats

    def start_workers(self):
        """Start all worker processes."""
        if self.processes_started:
            logger.warning("Workers already started")
            return

        try:
            logger.info("Starting worker processes...")

            if self.enable_hybrid and self.coordinator:
                # Start TokenizerManager workers
                for i in range(self.server_args.tokenizer_worker_num):
                    self._start_tokenizer_worker(i)

                # Start DetokenizerManager workers
                for i in range(self.server_args.detokenizer_processes):
                    self._start_detokenizer_worker(i)

                # Start Scheduler processes
                for i in range(self.server_args.dp_size):
                    self._start_scheduler_process(i)

                # Start coordination
                self.coordinator.start_coordination()

                self.processes_started = True
                logger.info("All worker processes started successfully")
            else:
                logger.info("Hybrid mode not enabled, skipping worker startup")

        except Exception as e:
            logger.error(f"Error starting workers: {e}")
            self.shutdown_workers()
            raise

    def _start_tokenizer_worker(self, worker_id: int):
        """Start a TokenizerManager worker process."""
        try:
            # Create port arguments for this worker
            port_args = PortArgs.init_new(self.server_args, dp_rank=worker_id)

            # Create and start process
            process = mp.Process(
                target=self._tokenizer_worker_main,
                args=(worker_id, port_args),
                name=f"TokenizerManager-{worker_id}",
            )
            process.start()

            self.tokenizer_workers[worker_id] = process
            logger.info(
                f"TokenizerManager worker {worker_id} started (PID: {process.pid})"
            )

        except Exception as e:
            logger.error(f"Error starting TokenizerManager worker {worker_id}: {e}")
            raise

    def _start_detokenizer_worker(self, worker_id: int):
        """Start a DetokenizerManager worker process."""
        try:
            # Create port arguments for this worker
            port_args = PortArgs.init_new(self.server_args, dp_rank=worker_id)

            # Create and start process
            process = mp.Process(
                target=self._detokenizer_worker_main,
                args=(worker_id, port_args),
                name=f"DetokenizerManager-{worker_id}",
            )
            process.start()

            self.detokenizer_workers[worker_id] = process
            logger.info(
                f"DetokenizerManager worker {worker_id} started (PID: {process.pid})"
            )

        except Exception as e:
            logger.error(f"Error starting DetokenizerManager worker {worker_id}: {e}")
            raise

    def _start_scheduler_process(self, rank: int):
        """Start a Scheduler process."""
        try:
            # Create port arguments for this scheduler
            port_args = PortArgs.init_new(self.server_args, dp_rank=rank)

            # Create and start process
            process = mp.Process(
                target=self._scheduler_main,
                args=(rank, port_args),
                name=f"Scheduler-{rank}",
            )
            process.start()

            self.scheduler_processes[rank] = process
            logger.info(f"Scheduler process {rank} started (PID: {process.pid})")

        except Exception as e:
            logger.error(f"Error starting Scheduler process {rank}: {e}")
            raise

    def _tokenizer_worker_main(self, worker_id: int, port_args: PortArgs):
        """Main function for TokenizerManager worker process."""
        try:
            # Create enhanced TokenizerManager
            tokenizer_manager = EnhancedTokenizerManager(
                self.server_args, port_args, worker_id=worker_id
            )

            # Register with coordinator (this will be done via shared memory)
            logger.info(f"TokenizerManager worker {worker_id} initialized")

            # Keep process alive
            while not self.shutdown_event.is_set():
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error in TokenizerManager worker {worker_id}: {e}")
            sys.exit(1)

    def _detokenizer_worker_main(self, worker_id: int, port_args: PortArgs):
        """Main function for DetokenizerManager worker process."""
        try:
            # Create enhanced DetokenizerManager
            detokenizer_manager = EnhancedDetokenizerManager(
                self.server_args, port_args, worker_id=worker_id
            )

            # Register with coordinator (this will be done via shared memory)
            logger.info(f"DetokenizerManager worker {worker_id} initialized")

            # Keep process alive
            while not self.shutdown_event.is_set():
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error in DetokenizerManager worker {worker_id}: {e}")
            sys.exit(1)

    def _scheduler_main(self, rank: int, port_args: PortArgs):
        """Main function for Scheduler process."""
        try:
            # Create Scheduler
            scheduler = Scheduler(self.server_args, port_args, rank=rank)

            logger.info(f"Scheduler process {rank} initialized")

            # Keep process alive
            while not self.shutdown_event.is_set():
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error in Scheduler process {rank}: {e}")
            sys.exit(1)

    def shutdown_workers(self):
        """Shutdown all worker processes."""
        try:
            logger.info("Shutting down worker processes...")

            # Set shutdown event
            self.shutdown_event.set()

            # Stop coordination
            if self.coordinator:
                self.coordinator.stop_coordination()

            # Terminate processes
            for name, process in self.tokenizer_workers.items():
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                logger.info(f"TokenizerManager worker {name} terminated")

            for name, process in self.detokenizer_workers.items():
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                logger.info(f"DetokenizerManager worker {name} terminated")

            for name, process in self.scheduler_processes.items():
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                logger.info(f"Scheduler process {name} terminated")

            # Shutdown coordinator
            if self.coordinator:
                self.coordinator.shutdown()

            self.processes_started = False
            logger.info("All worker processes shut down successfully")

        except Exception as e:
            logger.error(f"Error shutting down workers: {e}")

    def run(self, host: str = None, port: int = None):
        """Run the enhanced HTTP server."""
        try:
            # Start workers
            self.start_workers()

            # Get host and port
            host = host or self.server_args.host
            port = port or self.server_args.port

            # Setup signal handlers
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, shutting down...")
                self.shutdown_workers()
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            # Start server
            logger.info(f"Starting enhanced HTTP server on {host}:{port}")
            uvicorn.run(
                self.app, host=host, port=port, log_level=self.server_args.log_level
            )

        except Exception as e:
            logger.error(f"Error running enhanced HTTP server: {e}")
            self.shutdown_workers()
            raise
        finally:
            self.shutdown_workers()


def launch_enhanced_server(server_args: ServerArgs):
    """Launch the enhanced HTTP server."""
    try:
        server = EnhancedHTTPServer(server_args)
        server.run()
    except Exception as e:
        logger.error(f"Failed to launch enhanced server: {e}")
        raise


if __name__ == "__main__":
    # This would be called from the main entry point
    pass
