"""
Minimal HTTP load balancer for prefill and decode servers for testing.

Features:
- Request completion tracking with timing information
- Detailed logging for debugging request flow
- Support for both streaming and non-streaming responses
- Health checks and server registration
"""

import asyncio
import dataclasses
import logging
import random
import time
import urllib
from itertools import chain
from typing import Dict, List, Optional

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sglang.srt.disaggregation.utils import PDRegistryRequest
from sglang.srt.utils import maybe_wrap_ipv6_address

AIOHTTP_STREAM_READ_CHUNK_SIZE = (
    1024 * 64
)  # 64KB, to prevent aiohttp's "Chunk too big" error


def setup_logger(
    log_level="INFO",
    log_file=None,
    log_format="detailed",
    enable_request_logging=True,
    enable_debug_logging=False,
):
    logger = logging.getLogger("pdlb")

    # Set log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logger.setLevel(numeric_level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter based on format choice
    if log_format == "detailed":
        formatter = logging.Formatter(
            "[PDLB (Python)] %(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    elif log_format == "simple":
        formatter = logging.Formatter("[PDLB] %(levelname)s - %(message)s")
    elif log_format == "json":
        import json

        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "logger": "pdlb",
                    "message": record.getMessage(),
                }
                if hasattr(record, "funcName"):
                    log_entry["function"] = record.funcName
                if hasattr(record, "lineno"):
                    log_entry["line"] = record.lineno
                return json.dumps(log_entry)

        formatter = JSONFormatter()
    else:
        raise ValueError(f"Invalid log format: {log_format}")

    # Create handler
    if log_file:
        handler = logging.FileHandler(log_file)
        logger.info(f"Logging to file: {log_file}")
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Set global flags for conditional logging
    global ENABLE_REQUEST_LOGGING, ENABLE_DEBUG_LOGGING
    ENABLE_REQUEST_LOGGING = enable_request_logging
    ENABLE_DEBUG_LOGGING = enable_debug_logging

    logger.info(
        f"Logger initialized with level={log_level}, format={log_format}, file={log_file}"
    )
    logger.info(
        f"Request logging: {'enabled' if enable_request_logging else 'disabled'}"
    )
    logger.info(f"Debug logging: {'enabled' if enable_debug_logging else 'disabled'}")

    return logger


# Global request counter for tracking
request_counter = 0

# Global flags for conditional logging
ENABLE_REQUEST_LOGGING = True
ENABLE_DEBUG_LOGGING = False

# Global streaming request tracking
streaming_requests: Dict[int, Dict[str, any]] = {}
streaming_requests_lock = asyncio.Lock()

logger = setup_logger()  # Initialize with defaults


async def track_streaming_request_start(
    request_id: int, request_data: dict, prefill_server: str, decode_server: str
):
    """Track when a streaming request starts."""
    async with streaming_requests_lock:
        streaming_requests[request_id] = {
            "start_time": time.time(),
            "request_data": request_data,
            "prefill_server": prefill_server,
            "decode_server": decode_server,
            "streaming_started": False,
            "streaming_completed": False,
            "completion_time": None,
            "total_duration": None,
        }
        logger.info(
            f"🚀 Streaming request #{request_id} tracking started - prefill: {prefill_server}, decode: {decode_server}"
        )


async def track_streaming_started(request_id: int):
    """Track when streaming actually begins."""
    async with streaming_requests_lock:
        if request_id in streaming_requests:
            streaming_requests[request_id]["streaming_started"] = True
            streaming_start_time = time.time()
            time_to_streaming = (
                streaming_start_time - streaming_requests[request_id]["start_time"]
            )
            logger.info(
                f"📡 Streaming request #{request_id} started streaming after {time_to_streaming:.3f}s"
            )


async def track_streaming_completion(request_id: int):
    """Track when streaming completes."""
    async with streaming_requests_lock:
        if request_id in streaming_requests:
            completion_time = time.time()
            streaming_requests[request_id]["streaming_completed"] = True
            streaming_requests[request_id]["completion_time"] = completion_time
            streaming_requests[request_id]["total_duration"] = (
                completion_time - streaming_requests[request_id]["start_time"]
            )

            total_duration = streaming_requests[request_id]["total_duration"]
            logger.info(
                f"✅ Streaming request #{request_id} completed in {total_duration:.3f}s"
            )

            # Clean up completed request tracking
            del streaming_requests[request_id]


def get_streaming_stats() -> Dict[str, any]:
    """Get statistics about streaming requests."""
    with asyncio.Lock():
        active_streaming = len(streaming_requests)
        completed_requests = []

        for req_id, req_info in streaming_requests.items():
            if req_info["streaming_completed"]:
                completed_requests.append(
                    {
                        "request_id": req_id,
                        "total_duration": req_info["total_duration"],
                        "prefill_server": req_info["prefill_server"],
                        "decode_server": req_info["decode_server"],
                    }
                )

        return {
            "active_streaming_requests": active_streaming,
            "completed_streaming_requests": len(completed_requests),
            "streaming_requests_details": list(streaming_requests.keys()),
        }


async def clear_completed_streaming_requests():
    """Clear completed streaming requests from tracking."""
    async with streaming_requests_lock:
        completed_ids = [
            req_id
            for req_id, req_info in streaming_requests.items()
            if req_info["streaming_completed"]
        ]

        for req_id in completed_ids:
            del streaming_requests[req_id]

        if completed_ids:
            logger.info(
                f"Cleared {len(completed_ids)} completed streaming requests from tracking"
            )

        return len(completed_ids)


def get_streaming_request_details(request_id: int) -> Optional[Dict[str, any]]:
    """Get detailed information about a specific streaming request."""
    if request_id in streaming_requests:
        req_info = streaming_requests[request_id].copy()
        current_time = time.time()

        # Add calculated fields
        if req_info["start_time"]:
            req_info["age"] = current_time - req_info["start_time"]

        if req_info["streaming_started"] and req_info["start_time"]:
            req_info["time_to_streaming"] = (
                req_info.get("streaming_start_time", 0) - req_info["start_time"]
            )

        return req_info

    return None


@dataclasses.dataclass
class PrefillConfig:
    url: str
    bootstrap_port: Optional[int] = None


class MiniLoadBalancer:
    def __init__(
        self,
        prefill_configs: List[PrefillConfig],
        decode_servers: List[str],
        policy: str = "random",
    ):
        self.prefill_configs = prefill_configs
        self.prefill_servers = [p.url for p in prefill_configs]
        self.decode_servers = decode_servers
        self.policy = policy
        # Initialize round-robin counters
        self.prefill_counter = 0
        self.decode_counter = 0
        logger.info(
            f"MiniLoadBalancer initialized with {len(self.prefill_configs)} prefill servers and {len(self.decode_servers)} decode servers using {policy} policy"
        )

    def add_prefill_server(self, new_prefill_config: PrefillConfig):
        self.prefill_configs.append(new_prefill_config)
        self.prefill_servers.append(new_prefill_config.url)
        # Reset counter to ensure fair distribution with new server
        self.prefill_counter = 0
        logger.info(
            f"Added prefill server: {new_prefill_config.url} (total: {len(self.prefill_servers)})"
        )

    def add_decode_server(self, new_decode_server: str):
        self.decode_servers.append(new_decode_server)
        # Reset counter to ensure fair distribution with new server
        self.decode_counter = 0
        logger.info(
            f"Added decode server: {new_decode_server} (total: {len(self.decode_servers)})"
        )

    def get_streaming_stats(self) -> Dict[str, any]:
        """Get streaming request statistics."""
        return get_streaming_stats()

    def get_round_robin_state(self) -> Dict[str, any]:
        """Get current round-robin state for debugging."""
        return {
            "policy": self.policy,
            "prefill_counter": self.prefill_counter,
            "decode_counter": self.decode_counter,
            "prefill_servers_count": len(self.prefill_servers),
            "decode_servers_count": len(self.decode_servers),
            "next_prefill_index": (
                self.prefill_counter % len(self.prefill_servers)
                if self.prefill_servers
                else None
            ),
            "next_decode_index": (
                self.decode_counter % len(self.decode_servers)
                if self.decode_servers
                else None
            ),
        }

    def get_policy(self) -> str:
        """Get current load balancing policy."""
        return self.policy

    def select_pair(self):
        # TODO: return some message instead of panic
        assert len(self.prefill_configs) > 0, "No prefill servers available"
        assert len(self.decode_servers) > 0, "No decode servers available"

        if self.policy == "round_robin":
            # Round-robin selection for prefill servers
            prefill_config = self.prefill_configs[
                self.prefill_counter % len(self.prefill_configs)
            ]
            self.prefill_counter += 1

            # Round-robin selection for decode servers
            decode_server = self.decode_servers[
                self.decode_counter % len(self.decode_servers)
            ]
            self.decode_counter += 1
        else:  # random policy
            # Random selection for prefill servers
            prefill_config = random.choice(self.prefill_configs)
            # Random selection for decode servers
            decode_server = random.choice(self.decode_servers)

        if ENABLE_DEBUG_LOGGING:
            logger.debug(
                f"Selected prefill: {prefill_config.url}, decode: {decode_server} using {self.policy} policy"
            )

        # Log policy usage for monitoring
        if self.policy == "round_robin":
            logger.debug(
                f"Round-robin selection - prefill counter: {self.prefill_counter-1}, decode counter: {self.decode_counter-1}"
            )
        else:
            logger.debug(f"Random selection completed")

        return prefill_config.url, prefill_config.bootstrap_port, decode_server

    async def generate(
        self, modified_request, prefill_server, decode_server, endpoint
    ) -> ORJSONResponse:
        request_start_time = time.time()
        logger.info(f"Starting non-streaming generation via {endpoint}")
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=3600
            )  # Add timeout for request reliability
        ) as session:
            if ENABLE_DEBUG_LOGGING:
                logger.debug(
                    f"Sending requests to prefill: {prefill_server}/{endpoint}, decode: {decode_server}/{endpoint}"
                )
            tasks = [
                session.post(f"{prefill_server}/{endpoint}", json=modified_request),
                session.post(f"{decode_server}/{endpoint}", json=modified_request),
            ]

            # Wait for both responses to complete. Prefill should end first.
            try:
                prefill_response, decode_response = await asyncio.gather(*tasks)
                if ENABLE_DEBUG_LOGGING:
                    logger.debug(
                        f"Both responses received - prefill: {prefill_response.status}, decode: {decode_response.status}"
                    )
            except Exception as e:
                request_duration = time.time() - request_start_time
                logger.error(
                    f"Error during parallel requests after {request_duration:.3f}s: {e}"
                )
                raise

            if "return_logprob" in modified_request:
                if ENABLE_DEBUG_LOGGING:
                    logger.debug("Processing logprob merge")
                prefill_json = await prefill_response.json()
                ret_json = await decode_response.json()

                # merge `meta_info.input_token_logprobs` from prefill to decode
                if "meta_info" in ret_json:
                    if "input_token_logprobs" in ret_json["meta_info"]:
                        ret_json["meta_info"]["input_token_logprobs"] = (
                            prefill_json["meta_info"]["input_token_logprobs"]
                            + ret_json["meta_info"]["input_token_logprobs"]
                        )
            else:
                ret_json = await decode_response.json()

            request_duration = time.time() - request_start_time
            logger.info(
                f"Generation completed successfully via {endpoint} in {request_duration:.3f}s"
            )
            return ORJSONResponse(
                content=ret_json,
                status_code=decode_response.status,
            )

    async def generate_stream(
        self,
        modified_request,
        prefill_server,
        decode_server,
        endpoint="generate",
        request_id: Optional[int] = None,
    ):
        request_start_time = time.time()
        logger.info(f"Starting streaming generation via {endpoint}")
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        # Track streaming request if ID provided
        if request_id is not None:
            await track_streaming_request_start(
                request_id, modified_request, prefill_server, decode_server
            )

        async def stream_results():
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(
                        total=3600
                    )  # Add timeout for request reliability
                ) as session:
                    if ENABLE_DEBUG_LOGGING:
                        logger.debug(
                            f"Sending streaming requests to prefill: {prefill_server}/{endpoint}, decode: {decode_server}/{endpoint}"
                        )
                    # Create the tasks for both prefill and decode requests
                    tasks = [
                        session.post(
                            f"{prefill_server}/{endpoint}", json=modified_request
                        ),
                        session.post(
                            f"{decode_server}/{endpoint}", json=modified_request
                        ),
                    ]
                    # Wait for both responses to complete. Since this is streaming, they return immediately.
                    try:
                        prefill_response, decode_response = await asyncio.gather(*tasks)
                        if ENABLE_DEBUG_LOGGING:
                            logger.debug(
                                f"Streaming responses received - prefill: {prefill_response.status}, decode: {decode_response.status}"
                            )

                        # Track when streaming actually starts
                        if request_id is not None:
                            await track_streaming_started(request_id)

                    except Exception as e:
                        request_duration = time.time() - request_start_time
                        logger.error(
                            f"Error during streaming parallel requests after {request_duration:.3f}s: {e}"
                        )
                        raise

                    if modified_request.get("return_logprob", False):
                        if ENABLE_DEBUG_LOGGING:
                            logger.debug("Processing streaming logprob merge")
                        prefill_chunks = []
                        async for chunk in prefill_response.content:
                            prefill_chunks.append(chunk)

                        first_prefill_chunk = (
                            prefill_chunks[0].decode("utf-8")[5:].strip("\n")
                        )
                        first_prefill_chunk_json = orjson.loads(first_prefill_chunk)

                        async for chunk in decode_response.content:
                            # Note: This is inefficient
                            # merge prefill input_token_logprobs, output_token_logprobs to decode
                            decoded_chunk = chunk.decode("utf-8")
                            if (
                                decoded_chunk
                                and decoded_chunk.startswith("data:")
                                and "[DONE]" not in decoded_chunk
                            ):
                                ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))
                                ret_json["meta_info"]["input_token_logprobs"] = (
                                    first_prefill_chunk_json["meta_info"][
                                        "input_token_logprobs"
                                    ]
                                    + ret_json["meta_info"]["input_token_logprobs"]
                                )

                                yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                            else:
                                yield chunk
                    else:
                        if ENABLE_DEBUG_LOGGING:
                            logger.debug("Streaming decode response without logprob")
                        async for chunk in decode_response.content.iter_chunked(
                            AIOHTTP_STREAM_READ_CHUNK_SIZE
                        ):
                            yield chunk
            finally:
                # Log completion when streaming ends
                request_duration = time.time() - request_start_time
                logger.info(
                    f"Streaming generation completed via {endpoint} in {request_duration:.3f}s"
                )

                # Track streaming completion
                if request_id is not None:
                    await track_streaming_completion(request_id)

        logger.info(f"Streaming generation setup completed via {endpoint}")
        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )


app = FastAPI()
load_balancer: Optional[MiniLoadBalancer] = None


@app.get("/health")
async def health_check():
    logger.debug("Health check endpoint called")
    return Response(status_code=200)


@app.get("/health_generate")
async def health_check():
    logger.info("Health generate endpoint called")
    if load_balancer is None:
        logger.error("Load balancer not initialized for health check")
        raise HTTPException(status_code=500, detail="Load balancer not ready")

    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    logger.info(
        f"Checking health of {len(prefill_servers)} prefill and {len(decode_servers)} decode servers"
    )

    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(prefill_servers, decode_servers):
            tasks.append(session.post(f"{server}/health_generate"))

        try:
            for i, response in enumerate(asyncio.as_completed(tasks)):
                await response
            logger.info("All servers responded to health check")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

    return Response(status_code=200)


@app.post("/flush_cache")
async def flush_cache():
    logger.info("Flush cache endpoint called")
    if load_balancer is None:
        logger.error("Load balancer not initialized for flush cache")
        raise HTTPException(status_code=500, detail="Load balancer not ready")

    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    logger.info(
        f"Flushing cache on {len(prefill_servers)} prefill and {len(decode_servers)} decode servers"
    )

    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(prefill_servers, decode_servers):
            tasks.append(session.post(f"{server}/flush_cache"))

        try:
            for i, response in enumerate(asyncio.as_completed(tasks)):
                await response
            logger.info("Cache flush completed for all servers")
        except Exception as e:
            logger.error(f"Cache flush failed: {e}")
            raise HTTPException(status_code=500, detail=f"Cache flush failed: {e}")

    return Response(status_code=200)


@app.get("/get_server_info")
async def get_server_info():
    logger.debug("Get server info endpoint called")
    if load_balancer is None:
        logger.error("Load balancer not initialized for get server info")
        raise HTTPException(status_code=500, detail="Load balancer not ready")

    prefill_servers, decode_servers = (
        load_balancer.prefill_servers,
        load_balancer.decode_servers,
    )
    prefill_infos = []
    decode_infos = []
    all_internal_states = []

    async with aiohttp.ClientSession() as session:
        if ENABLE_DEBUG_LOGGING:
            logger.debug(
                f"Fetching server info from {len(prefill_servers)} prefill servers"
            )
        for server in chain(prefill_servers):
            try:
                server_info = await session.get(f"{server}/get_server_info")
                prefill_infos.append(await server_info.json())
            except Exception as e:
                logger.error(f"Failed to get info from prefill server {server}: {e}")

        if ENABLE_DEBUG_LOGGING:
            logger.debug(
                f"Fetching server info from {len(decode_servers)} decode servers"
            )
        for server in chain(decode_servers):
            try:
                server_info = await session.get(f"{server}/get_server_info")
                info_json = await server_info.json()
                decode_infos.append(info_json)
                # Extract internal_states from decode servers
                if "internal_states" in info_json:
                    all_internal_states.extend(info_json["internal_states"])
            except Exception as e:
                logger.error(f"Failed to get info from decode server {server}: {e}")

    # Return format expected by bench_one_batch_server.py
    if all_internal_states:
        if ENABLE_DEBUG_LOGGING:
            logger.debug(
                f"Returning server info with {len(all_internal_states)} internal states"
            )
        return {
            "internal_states": all_internal_states,
            "prefill": prefill_infos,
            "decode": decode_infos,
        }
    else:
        # Fallback with dummy data if no internal states found
        logger.warning("No internal states found, returning dummy data")
        return {
            "internal_states": [
                {
                    "last_gen_throughput": 0.0,
                    "avg_spec_accept_length": None,
                }
            ],
            "prefill": prefill_infos,
            "decode": decode_infos,
        }


@app.get("/get_streaming_stats")
async def get_streaming_stats():
    """Get statistics about streaming requests."""
    logger.debug("Get streaming stats endpoint called")

    stats = get_streaming_stats()
    logger.info(f"Streaming stats: {stats}")

    return {"streaming_stats": stats, "timestamp": time.time()}


@app.get("/get_round_robin_state")
async def get_round_robin_state():
    """Get current round-robin state for debugging."""
    logger.debug("Get round-robin state endpoint called")

    if load_balancer is None:
        logger.error("Load balancer not initialized for get round-robin state")
        raise HTTPException(status_code=500, detail="Load balancer not ready")

    try:
        state = load_balancer.get_round_robin_state()
        logger.info(f"Round-robin state: {state}")

        return {"round_robin_state": state, "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Failed to get round-robin state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_policy")
async def get_policy():
    """Get current load balancing policy."""
    logger.debug("Get policy endpoint called")

    if load_balancer is None:
        logger.error("Load balancer not initialized for get policy")
        raise HTTPException(status_code=500, detail="Load balancer not ready")

    try:
        policy = load_balancer.get_policy()
        logger.info(f"Current policy: {policy}")

        return {"policy": policy, "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Failed to get policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear_streaming_requests")
async def clear_streaming_requests():
    """Clear completed streaming requests from tracking."""
    logger.debug("Clear streaming requests endpoint called")

    if load_balancer is None:
        logger.error("Load balancer not initialized for clear streaming requests")
        raise HTTPException(status_code=500, detail="Load balancer not ready")

    try:
        cleared_count = await clear_completed_streaming_requests()
        logger.info(f"Cleared {cleared_count} completed streaming requests")

        return {"cleared_count": cleared_count, "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Failed to clear streaming requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_streaming_request/{request_id}")
async def get_streaming_request_details(request_id: int):
    """Get detailed information about a specific streaming request."""
    logger.debug(f"Get streaming request details for request #{request_id}")

    if load_balancer is None:
        logger.error("Load balancer not initialized for get streaming request details")
        raise HTTPException(status_code=500, detail="Load balancer not ready")

    try:
        request_details = get_streaming_request_details(request_id)

        if request_details is None:
            raise HTTPException(
                status_code=404, detail=f"Streaming request #{request_id} not found"
            )

        logger.info(f"Retrieved details for streaming request #{request_id}")

        return {
            "request_id": request_id,
            "details": request_details,
            "timestamp": time.time(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get streaming request details for #{request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_model_info")
async def get_model_info():
    logger.debug("Get model info endpoint called")
    # Dummy model information
    model_info = {
        "model_path": "/path/to/dummy/model",
        "tokenizer_path": "/path/to/dummy/tokenizer",
        "is_generation": True,
        "preferred_sampling_params": {"temperature": 0.7, "max_new_tokens": 128},
    }
    return ORJSONResponse(content=model_info)


@app.post("/generate")
async def handle_generate_request(request_data: dict):
    global request_counter
    request_counter += 1
    request_start_time = time.time()

    if ENABLE_REQUEST_LOGGING:
        logger.info(f"POST /generate request #{request_counter} received")
        logger.info(f"Request data: {request_data}")
    else:
        logger.debug(f"POST /generate request #{request_counter} received")

    if load_balancer is None:
        logger.error("Load balancer not initialized")
        raise HTTPException(status_code=500, detail="Load balancer not ready")

    try:
        prefill_server, bootstrap_port, decode_server = load_balancer.select_pair()
        if ENABLE_REQUEST_LOGGING:
            logger.info(
                f"Selected servers - prefill: {prefill_server}, decode: {decode_server}, bootstrap_port: {bootstrap_port}"
            )
        else:
            logger.debug(
                f"Selected servers - prefill: {prefill_server}, decode: {decode_server}, bootstrap_port: {bootstrap_port}"
            )

        # Parse and transform prefill_server for bootstrap data
        parsed_url = urllib.parse.urlparse(prefill_server)
        hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
        modified_request = request_data.copy()

        batch_size = _get_request_batch_size(modified_request)
        if batch_size is not None:
            if ENABLE_DEBUG_LOGGING:
                logger.debug(f"Batch request detected with size: {batch_size}")
            modified_request.update(
                {
                    "bootstrap_host": [hostname] * batch_size,
                    "bootstrap_port": [bootstrap_port] * batch_size,
                    "bootstrap_room": [
                        _generate_bootstrap_room() for _ in range(batch_size)
                    ],
                }
            )
        else:
            if ENABLE_DEBUG_LOGGING:
                logger.debug("Single request detected")
            modified_request.update(
                {
                    "bootstrap_host": hostname,
                    "bootstrap_port": bootstrap_port,
                    "bootstrap_room": _generate_bootstrap_room(),
                }
            )

        if ENABLE_DEBUG_LOGGING:
            logger.debug(f"Modified request: {modified_request}")

        if request_data.get("stream", False):
            if ENABLE_REQUEST_LOGGING:
                logger.info(
                    f"Returning streaming response for request #{request_counter}"
                )
            response = await load_balancer.generate_stream(
                modified_request,
                prefill_server,
                decode_server,
                "generate",
                request_id=request_counter,
            )
            request_duration = time.time() - request_start_time
            logger.info(
                f"Request #{request_counter} streaming response returned in {request_duration:.3f}s"
            )
            return response
        else:
            if ENABLE_REQUEST_LOGGING:
                logger.info(
                    f"Returning non-streaming response for request #{request_counter}"
                )
            response = await load_balancer.generate(
                modified_request, prefill_server, decode_server, "generate"
            )
            request_duration = time.time() - request_start_time
            logger.info(
                f"Request #{request_counter} non-streaming response returned in {request_duration:.3f}s"
            )
            return response
    except Exception as e:
        request_duration = time.time() - request_start_time
        logger.error(
            f"Error in handle_generate_request #{request_counter} after {request_duration:.3f}s: {e}"
        )
        if ENABLE_REQUEST_LOGGING:
            logger.error(f"Request data that caused error: {request_data}")
        raise


async def _forward_to_backend(request_data: dict, endpoint_name: str):
    global request_counter
    request_counter += 1
    request_start_time = time.time()

    if ENABLE_REQUEST_LOGGING:
        logger.info(f"POST /{endpoint_name} request #{request_counter} received")
        logger.info(f"Request data: {request_data}")
    else:
        logger.debug(f"POST /{endpoint_name} request #{request_counter} received")

    if load_balancer is None:
        logger.error("Load balancer not initialized")
        raise HTTPException(status_code=500, detail="Load balancer not ready")

    try:
        prefill_server, bootstrap_port, decode_server = load_balancer.select_pair()
        if ENABLE_REQUEST_LOGGING:
            logger.info(
                f"Selected servers for {endpoint_name} - prefill: {prefill_server}, decode: {decode_server}"
            )
        else:
            logger.debug(
                f"Selected servers for {endpoint_name} - prefill: {prefill_server}, decode: {decode_server}"
            )

        # Parse and transform prefill_server for bootstrap data
        parsed_url = urllib.parse.urlparse(prefill_server)
        hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
        modified_request = request_data.copy()
        modified_request.update(
            {
                "bootstrap_host": hostname,
                "bootstrap_port": bootstrap_port,
                "bootstrap_room": _generate_bootstrap_room(),
            }
        )

        if ENABLE_DEBUG_LOGGING:
            logger.debug(f"Modified request for {endpoint_name}: {modified_request}")

        if request_data.get("stream", False):
            if ENABLE_REQUEST_LOGGING:
                logger.info(
                    f"Returning streaming response for {endpoint_name} request #{request_counter}"
                )
            response = await load_balancer.generate_stream(
                modified_request,
                prefill_server,
                decode_server,
                endpoint=endpoint_name,
                request_id=request_counter,
            )
            request_duration = time.time() - request_start_time
            logger.info(
                f"Request #{request_counter} {endpoint_name} streaming response returned in {request_duration:.3f}s"
            )
            return response
        else:
            if ENABLE_REQUEST_LOGGING:
                logger.info(
                    f"Returning non-streaming response for {endpoint_name} request #{request_counter}"
                )
            response = await load_balancer.generate(
                modified_request,
                prefill_server,
                decode_server,
                endpoint=endpoint_name,
            )
            request_duration = time.time() - request_start_time
            logger.info(
                f"Request #{request_counter} {endpoint_name} non-streaming response returned in {request_duration:.3f}s"
            )
            return response
    except Exception as e:
        request_duration = time.time() - request_start_time
        logger.error(
            f"Error in _forward_to_backend for {endpoint_name} request #{request_counter} after {request_duration:.3f}s: {e}"
        )
        if ENABLE_REQUEST_LOGGING:
            logger.error(f"Request data that caused error: {request_data}")
        raise


@app.post("/v1/chat/completions")
async def handle_chat_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/chat/completions")


@app.post("/v1/completions")
async def handle_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/completions")


def _generate_bootstrap_room():
    room_id = random.randint(0, 2**63 - 1)
    if ENABLE_DEBUG_LOGGING:
        logger.debug(f"Generated bootstrap room ID: {room_id}")
    return room_id


# We may utilize `GenerateReqInput`'s logic later
def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        batch_size = None if isinstance(text, str) else len(text)
        if ENABLE_DEBUG_LOGGING:
            logger.debug(f"Batch size from text: {batch_size}")
        return batch_size
    if (input_ids := request.get("input_ids")) is not None:
        batch_size = None if isinstance(input_ids[0], int) else len(input_ids)
        if ENABLE_DEBUG_LOGGING:
            logger.debug(f"Batch size from input_ids: {batch_size}")
        return batch_size
    if ENABLE_DEBUG_LOGGING:
        logger.debug("No batch size detected")
    return None


@app.get("/v1/models")
async def get_models():
    logger.debug("Get models endpoint called")
    if load_balancer is None:
        logger.error("Load balancer not initialized for get models")
        raise HTTPException(status_code=500, detail="Load balancer not ready")

    prefill_server = load_balancer.prefill_servers[0]  # Get the first prefill server
    logger.info(f"Fetching models from prefill server: {prefill_server}")

    async with aiohttp.ClientSession() as session:
        try:
            response = await session.get(f"{prefill_server}/v1/models")
            if response.status != 200:
                logger.error(f"Prefill server error: Status {response.status}")
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Prefill server error: Status {response.status}",
                )
            models = await response.json()
            logger.info(f"Successfully retrieved models from prefill server")
            return ORJSONResponse(content=models)
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/register")
async def register(obj: PDRegistryRequest):
    logger.info(
        f"Registration request received: mode={obj.mode}, url={obj.registry_url}"
    )

    if load_balancer is None:
        logger.error("Load balancer not initialized for registration")
        raise HTTPException(status_code=500, detail="Load balancer not ready")

    try:
        if obj.mode == "prefill":
            load_balancer.add_prefill_server(
                PrefillConfig(obj.registry_url, obj.bootstrap_port)
            )
            logger.info(
                f"Registered prefill server: {obj.registry_url} with bootstrap port: {obj.bootstrap_port}"
            )
        elif obj.mode == "decode":
            load_balancer.add_decode_server(obj.registry_url)
            logger.info(f"Registered decode server: {obj.registry_url}")
        else:
            logger.error(f"Invalid registration mode: {obj.mode}")
            raise HTTPException(
                status_code=400,
                detail="Invalid mode. Must be either PREFILL or DECODE.",
            )

        logger.info(
            f"Registration completed. #Prefill servers: {len(load_balancer.prefill_configs)}, "
            f"#Decode servers: {len(load_balancer.decode_servers)}"
        )

        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise


def run(
    prefill_configs,
    decode_addrs,
    host,
    port,
    policy="random",
    log_level="INFO",
    log_file=None,
    log_format="detailed",
    enable_request_logging=True,
    enable_debug_logging=False,
):
    global load_balancer, logger

    # Reconfigure logger with new settings
    logger = setup_logger(
        log_level, log_file, log_format, enable_request_logging, enable_debug_logging
    )

    logger.info(f"Starting mini load balancer on {host}:{port}")
    logger.info(f"Initial prefill configs: {prefill_configs}")
    logger.info(f"Initial decode addresses: {decode_addrs}")

    load_balancer = MiniLoadBalancer(prefill_configs, decode_addrs, policy)
    logger.info("Load balancer initialized successfully")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # FIXME: remove this, use the unified entry point: sglang.srt.disaggregation.launch_lb
    from sglang.srt.disaggregation.launch_lb import main

    main()
