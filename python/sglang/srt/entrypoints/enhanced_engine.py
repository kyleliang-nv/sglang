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
"""Enhanced engine with hybrid architecture support."""

import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.entrypoints.engine import Engine, _launch_subprocesses
from sglang.srt.entrypoints.enhanced_http_server import EnhancedHTTPServer
from sglang.srt.managers.enhanced_detokenizer_manager import EnhancedDetokenizerManager
from sglang.srt.managers.enhanced_tokenizer_manager import EnhancedTokenizerManager
from sglang.srt.managers.hybrid_coordinator import HybridCoordinator
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.server_args import PortArgs
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class EnhancedEngine(Engine):
    """Enhanced engine with hybrid architecture support."""

    def __init__(self, server_args: ServerArgs):
        super().__init__(server_args)

        # Check if we're in disaggregation mode
        self.disaggregation_mode = getattr(server_args, "disaggregation_mode", "null")
        self.is_disaggregated = self.disaggregation_mode != "null"

        # Only enable hybrid mode if not in disaggregation mode
        self.enable_hybrid = (
            server_args.enable_multi_tokenizer and not self.is_disaggregated
        )

        if self.is_disaggregated:
            logger.info(
                f"EnhancedEngine running in disaggregation mode: {self.disaggregation_mode}"
            )
            logger.info(
                "Hybrid architecture not supported in disaggregation mode, using standard architecture"
            )

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

        logger.info(
            f"EnhancedEngine initialized with hybrid mode: {self.enable_hybrid}"
        )

    def launch_hybrid_architecture(
        self,
    ) -> Tuple[EnhancedHTTPServer, HybridCoordinator]:
        """Launch the hybrid architecture with all worker processes."""
        if not self.enable_hybrid:
            raise RuntimeError("Hybrid mode not enabled")

        try:
            logger.info("🚀 Launching hybrid architecture...")

            # Start all worker processes
            self._start_hybrid_workers()

            # Create and start enhanced HTTP server
            http_server = EnhancedHTTPServer(self.server_args)

            # Start coordination
            self.coordinator.start_coordination()

            logger.info("✅ Hybrid architecture launched successfully")

            return http_server, self.coordinator

        except Exception as e:
            logger.error(f"❌ Failed to launch hybrid architecture: {e}")
            self._shutdown_hybrid_workers()
            raise

    def _start_hybrid_workers(self):
        """Start all hybrid architecture worker processes."""
        if self.processes_started:
            logger.warning("Hybrid workers already started")
            return

        try:
            logger.info("Starting hybrid architecture workers...")

            # Start TokenizerManager workers
            for i in range(self.server_args.tokenizer_worker_num):
                self._start_tokenizer_worker(i)

            # Start DetokenizerManager workers
            for i in range(self.server_args.detokenizer_processes):
                self._start_detokenizer_worker(i)

            # Start Scheduler processes
            for i in range(self.server_args.dp_size):
                self._start_scheduler_process(i)

            # Wait for workers to initialize
            self._wait_for_workers_ready()

            self.processes_started = True
            logger.info("All hybrid architecture workers started successfully")

        except Exception as e:
            logger.error(f"Error starting hybrid workers: {e}")
            self._shutdown_hybrid_workers()
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

    def _wait_for_workers_ready(self, timeout: int = 30):
        """Wait for all workers to be ready."""
        logger.info("Waiting for workers to be ready...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if all processes are alive
            all_alive = True

            for name, process in self.tokenizer_workers.items():
                if not process.is_alive():
                    all_alive = False
                    logger.warning(f"TokenizerManager worker {name} is not alive")

            for name, process in self.detokenizer_workers.items():
                if not process.is_alive():
                    all_alive = False
                    logger.warning(f"DetokenizerManager worker {name} is not alive")

            for name, process in self.scheduler_processes.items():
                if not process.is_alive():
                    all_alive = False
                    logger.warning(f"Scheduler process {name} is not alive")

            if all_alive:
                logger.info("All workers are ready")
                return

            time.sleep(1)

        raise RuntimeError(f"Workers not ready within {timeout} seconds")

    def _tokenizer_worker_main(self, worker_id: int, port_args: PortArgs):
        """Main function for TokenizerManager worker process."""
        try:
            # Set environment variables
            os.environ["SGLANG_WORKER_ID"] = str(worker_id)
            os.environ["SGLANG_WORKER_TYPE"] = "tokenizer"

            # Create enhanced TokenizerManager
            tokenizer_manager = EnhancedTokenizerManager(
                self.server_args, port_args, worker_id=worker_id
            )

            # Register with coordinator via shared memory
            logger.info(f"TokenizerManager worker {worker_id} initialized and ready")

            # Keep process alive
            while not self.shutdown_event.is_set():
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error in TokenizerManager worker {worker_id}: {e}")
            sys.exit(1)

    def _detokenizer_worker_main(self, worker_id: int, port_args: PortArgs):
        """Main function for DetokenizerManager worker process."""
        try:
            # Set environment variables
            os.environ["SGLANG_WORKER_ID"] = str(worker_id)
            os.environ["SGLANG_WORKER_TYPE"] = "detokenizer"

            # Create enhanced DetokenizerManager
            detokenizer_manager = EnhancedDetokenizerManager(
                self.server_args, port_args, worker_id=worker_id
            )

            # Register with coordinator via shared memory
            logger.info(f"DetokenizerManager worker {worker_id} initialized and ready")

            # Keep process alive
            while not self.shutdown_event.is_set():
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error in DetokenizerManager worker {worker_id}: {e}")
            sys.exit(1)

    def _scheduler_main(self, rank: int, port_args: PortArgs):
        """Main function for Scheduler process."""
        try:
            # Set environment variables
            os.environ["SGLANG_WORKER_ID"] = str(rank)
            os.environ["SGLANG_WORKER_TYPE"] = "scheduler"

            # Create Scheduler
            scheduler = Scheduler(self.server_args, port_args, rank=rank)

            logger.info(f"Scheduler process {rank} initialized and ready")

            # Keep process alive
            while not self.shutdown_event.is_set():
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error in Scheduler process {rank}: {e}")
            sys.exit(1)

    def _shutdown_hybrid_workers(self):
        """Shutdown all hybrid architecture worker processes."""
        try:
            logger.info("Shutting down hybrid architecture workers...")

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
            logger.info("All hybrid architecture workers shut down successfully")

        except Exception as e:
            logger.error(f"Error shutting down hybrid workers: {e}")

    def get_hybrid_stats(self) -> Dict[str, Any]:
        """Get statistics about the hybrid architecture."""
        if not self.enable_hybrid:
            return {"hybrid_mode": False}

        stats = {
            "hybrid_mode": True,
            "tokenizer_workers": len(self.tokenizer_workers),
            "detokenizer_workers": len(self.detokenizer_workers),
            "scheduler_processes": len(self.scheduler_processes),
            "processes_started": self.processes_started,
        }

        if self.coordinator:
            stats["coordinator_stats"] = self.coordinator.get_coordination_stats()

        return stats

    def shutdown(self):
        """Shutdown the enhanced engine."""
        try:
            logger.info("Shutting down EnhancedEngine...")

            # Shutdown hybrid workers if running
            if self.enable_hybrid and self.processes_started:
                self._shutdown_hybrid_workers()

            # Call parent shutdown
            super().shutdown()

            logger.info("EnhancedEngine shutdown complete")

        except Exception as e:
            logger.error(f"Error shutting down EnhancedEngine: {e}")


def launch_enhanced_engine(server_args: ServerArgs):
    """Launch the enhanced engine with hybrid architecture support."""
    try:
        # Check if we're in disaggregation mode
        disaggregation_mode = getattr(server_args, "disaggregation_mode", "null")
        is_disaggregated = disaggregation_mode != "null"

        if is_disaggregated:
            logger.info(f"Running in disaggregation mode: {disaggregation_mode}")
            logger.info("Using standard launch_server for disaggregation mode")
            # Import and use standard launch_server for disaggregation mode
            from sglang.srt.entrypoints.http_server import launch_server

            return launch_server(server_args), None, None

        # Create enhanced engine for non-disaggregated mode
        engine = EnhancedEngine(server_args)

        if server_args.enable_multi_tokenizer:
            # Launch hybrid architecture
            http_server, coordinator = engine.launch_hybrid_architecture()

            # Return the enhanced components
            return engine, http_server, coordinator
        else:
            # Launch standard architecture
            return engine, None, None

    except Exception as e:
        logger.error(f"Failed to launch enhanced engine: {e}")
        raise


def main():
    """Main entry point for the enhanced engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced SGLang Engine")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--model-path", type=str, required=True, help="Model path")
    parser.add_argument(
        "--tokenizer-worker-num",
        type=int,
        default=1,
        help="Number of TokenizerManager workers",
    )
    parser.add_argument(
        "--detokenizer-processes",
        type=int,
        default=1,
        help="Number of DetokenizerManager processes",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=30000, help="Port to bind to")

    args = parser.parse_args()

    # Create server arguments
    server_args = ServerArgs(
        model_path=args.model_path,
        tokenizer_worker_num=args.tokenizer_worker_num,
        detokenizer_processes=args.detokenizer_processes,
        host=args.host,
        port=args.port,
    )

    try:
        # Launch enhanced engine
        result = launch_enhanced_engine(server_args)

        # Check if we're in disaggregation mode
        disaggregation_mode = getattr(server_args, "disaggregation_mode", "null")
        is_disaggregated = disaggregation_mode != "null"

        if is_disaggregated:
            # In disaggregation mode, launch_server already handles everything
            logger.info(
                f"Standard launch_server completed for disaggregation mode: {disaggregation_mode}"
            )
            # The launch_server function doesn't return, so we shouldn't reach here
            return
        else:
            # Unpack the enhanced engine results
            engine, http_server, coordinator = result

            if http_server:
                # Run enhanced HTTP server with hybrid architecture
                http_server.run()
            else:
                # Run standard engine
                engine.run()

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Engine error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
