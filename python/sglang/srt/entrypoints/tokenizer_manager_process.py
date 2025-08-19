import asyncio
import json
import logging
import multiprocessing as mp
from typing import Any, Dict

import zmq

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.entrypoints.engine import _launch_subprocesses
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def run_tokenizer_manager_process(
    server_args: ServerArgs, ipc_port: int, pipe_writer: mp.connection.Connection = None
):
    """Run TokenizerManager in a separate process with ZMQ server"""

    try:
        # Check disaggregation mode
        disaggregation_mode = DisaggregationMode(server_args.disaggregation_mode)
        logger.info(f"Starting TokenizerManager process in {disaggregation_mode} mode")

        if disaggregation_mode == DisaggregationMode.NULL:
            # Full server mode - launch all subprocesses
            tokenizer_manager, template_manager, scheduler_info = _launch_subprocesses(
                server_args=server_args
            )
        elif disaggregation_mode == DisaggregationMode.DECODE:
            # Decode-only mode - only launch decode-related components
            logger.info("Running in decode-only mode - launching decode subprocesses")
            tokenizer_manager, template_manager, scheduler_info = _launch_subprocesses(
                server_args=server_args
            )
        elif disaggregation_mode == DisaggregationMode.PREFILL:
            # Prefill-only mode - launch prefill components
            logger.info("Running in prefill-only mode - launching prefill subprocesses")
            tokenizer_manager, template_manager, scheduler_info = _launch_subprocesses(
                server_args=server_args
            )
        else:
            raise ValueError(f"Unsupported disaggregation mode: {disaggregation_mode}")

        # Set up ZMQ server
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{ipc_port}")

        logger.info(
            f"TokenizerManager process started on port {ipc_port} in {disaggregation_mode} mode"
        )

        # Signal ready
        if pipe_writer:
            pipe_writer.send(
                {"status": "ready", "port": ipc_port, "mode": str(disaggregation_mode)}
            )

        # Main request handling loop
        while True:
            try:
                # Receive request
                request_data = socket.recv_json()
                request_type = request_data.get("type")

                if request_type == "generate":
                    # Handle generate request
                    response = handle_generate_request(
                        tokenizer_manager, request_data, disaggregation_mode
                    )
                elif request_type == "health":
                    # Handle health check
                    response = {"status": "healthy", "mode": str(disaggregation_mode)}
                elif request_type == "get_mode":
                    # Return disaggregation mode info
                    response = {"mode": str(disaggregation_mode), "status": "success"}
                else:
                    response = {"error": f"Unknown request type: {request_type}"}

                # Send response
                socket.send_json(response)

            except Exception as e:
                logger.error(f"Error handling request: {e}")
                socket.send_json({"error": str(e)})

    except Exception as e:
        logger.error(f"Failed to start TokenizerManager process: {e}")
        if pipe_writer:
            pipe_writer.send({"status": "error", "error": str(e)})
        raise


def handle_generate_request(
    tokenizer_manager: TokenizerManager,
    request_data: Dict[str, Any],
    disaggregation_mode: DisaggregationMode,
):
    """Handle a generate request from the HTTP server"""

    # Convert request data back to GenerateReqInput
    from sglang.srt.managers.io_struct import GenerateReqInput

    # Create the request object
    generate_req = GenerateReqInput(**request_data["data"])

    # Validate request for disaggregation mode
    if disaggregation_mode != DisaggregationMode.NULL:
        if (
            not hasattr(generate_req, "bootstrap_host")
            or not generate_req.bootstrap_host
        ):
            return {
                "status": "error",
                "error": f"Disaggregation mode {disaggregation_mode} requires bootstrap_host",
            }
        if (
            not hasattr(generate_req, "bootstrap_room")
            or generate_req.bootstrap_room is None
        ):
            return {
                "status": "error",
                "error": f"Disaggregation mode {disaggregation_mode} requires bootstrap_room",
            }

    # Call the tokenizer manager
    try:
        # Create a mock request object for the tokenizer manager
        class MockRequest:
            def __init__(self):
                self.headers = {}
                self.query_params = {}

        mock_request = MockRequest()

        # Run the async generate_request in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Get the generator
            generator = tokenizer_manager.generate_request(generate_req, mock_request)

            # Get the first (and only) response for non-streaming
            if not generate_req.stream:
                response = loop.run_until_complete(generator.__anext__())
                return {
                    "status": "success",
                    "data": response,
                    "mode": str(disaggregation_mode),
                }
            else:
                # For streaming, collect all responses
                responses = []
                try:
                    while True:
                        response = loop.run_until_complete(generator.__anext__())
                        responses.append(response)
                except StopAsyncIteration:
                    pass
                return {
                    "status": "success",
                    "data": responses,
                    "stream": True,
                    "mode": str(disaggregation_mode),
                }

        finally:
            loop.close()

    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    # For testing standalone
    import argparse

    parser = argparse.ArgumentParser()
    # Add your server args here
    args = parser.parse_args()

    # Create ServerArgs from command line
    server_args = ServerArgs()  # Configure as needed

    run_tokenizer_manager_process(server_args, ipc_port=31000)
