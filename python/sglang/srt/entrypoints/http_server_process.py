import asyncio
import json
import logging
import multiprocessing as mp
from typing import Any, Dict

import orjson
import zmq
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.entrypoints.http_server import app, lifespan
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class TokenizerClient:
    """Client for communicating with TokenizerManager process"""

    def __init__(self, endpoint: str):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{endpoint}")
        self.socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 second timeout
        self.disaggregation_mode = None

    def get_mode(self) -> Dict[str, Any]:
        """Get the disaggregation mode from TokenizerManager"""
        try:
            message = {"type": "get_mode"}
            self.socket.send_json(message)
            response = self.socket.recv_json()
            if response.get("status") == "success":
                self.disaggregation_mode = response.get("mode")
            return response
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def generate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send generate request to TokenizerManager"""
        try:
            message = {"type": "generate", "data": request_data}

            self.socket.send_json(message)
            response = self.socket.recv_json()
            return response

        except Exception as e:
            logger.error(f"Error communicating with TokenizerManager: {e}")
            return {"status": "error", "error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """Check health of TokenizerManager"""
        try:
            message = {"type": "health"}
            self.socket.send_json(message)
            response = self.socket.recv_json()
            return response
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def close(self):
        """Clean up ZMQ resources"""
        self.socket.close()
        self.context.term()


def run_http_server_process(
    server_args: ServerArgs,
    tokenizer_endpoint: str,
    pipe_writer: mp.connection.Connection = None,
):
    """Run HTTP server in separate process"""

    try:
        # Create TokenizerClient
        tokenizer_client = TokenizerClient(tokenizer_endpoint)

        # Get disaggregation mode from TokenizerManager
        mode_response = tokenizer_client.get_mode()
        if mode_response.get("status") == "success":
            disaggregation_mode = mode_response.get("mode")
            logger.info(f"HTTP server running in {disaggregation_mode} mode")
        else:
            logger.warning(f"Could not determine disaggregation mode: {mode_response}")
            disaggregation_mode = "unknown"

        # Override the generate endpoint to use TokenizerClient
        @app.post("/generate")
        async def generate_request(request: Request):
            """Handle generate request by forwarding to TokenizerManager"""
            try:
                # Parse request body
                body = await request.json()

                # Validate request for disaggregation mode
                if disaggregation_mode != "null":
                    if "bootstrap_host" not in body:
                        return {
                            "error": f"Disaggregation mode {disaggregation_mode} requires bootstrap_host"
                        }
                    if "bootstrap_room" not in body:
                        return {
                            "error": f"Disaggregation mode {disaggregation_mode} requires bootstrap_room"
                        }

                # Forward to TokenizerManager
                response = tokenizer_client.generate_request(body)

                if response.get("status") == "success":
                    return response["data"]
                else:
                    return {"error": response.get("error", "Unknown error")}

            except Exception as e:
                logger.error(f"Error in generate_request: {e}")
                return {"error": str(e)}

        # Override health check
        @app.get("/health_generate")
        async def health_generate():
            """Check health by communicating with TokenizerManager"""
            health = tokenizer_client.health_check()
            if health.get("status") == "healthy":
                return Response(status_code=200)
            else:
                return Response(status_code=503)

        # Add mode info endpoint
        @app.get("/get_mode")
        async def get_mode():
            """Get disaggregation mode information"""
            return {"mode": disaggregation_mode, "status": "success"}

        # Signal ready
        if pipe_writer:
            pipe_writer.send(
                {
                    "status": "ready",
                    "endpoint": tokenizer_endpoint,
                    "mode": disaggregation_mode,
                }
            )

        logger.info(
            f"HTTP server started, connected to TokenizerManager at {tokenizer_endpoint} in {disaggregation_mode} mode"
        )

        # Start the server
        import uvicorn

        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level_http or server_args.log_level,
        )

    except Exception as e:
        logger.error(f"Failed to start HTTP server process: {e}")
        if pipe_writer:
            pipe_writer.send({"status": "error", "error": str(e)})
        raise
    finally:
        if "tokenizer_client" in locals():
            tokenizer_client.close()


if __name__ == "__main__":
    # For testing standalone
    import argparse

    parser = argparse.ArgumentParser()
    # Add your server args here
    args = parser.parse_args()

    # Create ServerArgs from command line
    server_args = ServerArgs()  # Configure as needed

    run_http_server_process(server_args, tokenizer_endpoint="localhost:31000")
