"""Launch the inference server."""

import os
import sys

from sglang.srt.entrypoints.enhanced_engine import launch_enhanced_engine
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        # Use enhanced engine with hybrid architecture support
        engine, http_server, coordinator = launch_enhanced_engine(server_args)

        if http_server:
            # Run enhanced HTTP server with hybrid architecture
            http_server.run()
        else:
            # Fall back to standard engine if hybrid mode not enabled
            engine.run()
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
