"""Launch the inference server."""

import os
import sys

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    # Check if hybrid mode is enabled
    if (
        hasattr(server_args, "enable_multi_tokenizer")
        and server_args.enable_multi_tokenizer
    ):
        print(
            f"🚀 Hybrid architecture enabled with {server_args.tokenizer_worker_num} TM workers and {server_args.detokenizer_processes} DM processes"
        )
        print(
            f"📊 Load balancing policy: {server_args.detokenizer_load_balance_policy}"
        )
        print(
            "⚠️  Note: Hybrid architecture is not yet fully implemented for disaggregation mode"
        )
        print("   Using standard architecture with hybrid arguments ignored")

    try:
        # Use standard launch_server for now
        # TODO: Implement full hybrid architecture support
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
