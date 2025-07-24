# MODIFIED FROM tuning_deepep.py for PyTorch distributed communication

"""
Example usage for PyTorch distributed communication:
python tuning_pytorch_distributed.py --nnodes 1 --node-rank 0 --master-addr 127.0.0.1
Then check `deepep_tuned_pytorch_dist.json`
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
from deepep_utils import (
    bench,
    create_grouped_scores,
    init_dist,
    inplace_unique,
    per_token_cast_to_fp8,
)


def test_main(
    num_sms: int,
    local_rank: int,
    num_local_ranks: int,
    num_ranks: int,
    num_nodes: int,
    rank: int,
    group: dist.ProcessGroup,
    args,
):
    # Settings - simplified for PyTorch distributed
    num_tokens, hidden, num_topk_groups, num_topk, num_experts = (
        4096,
        7168,
        min(num_nodes, 4),
        8,
        (256 // num_ranks) * num_ranks,
    )
    assert num_experts % num_ranks == 0 and num_local_ranks == 8
    if local_rank == 0:
        print(
            f"[config] num_tokens={num_tokens}, hidden={hidden}, num_topk_groups={num_topk_groups}, num_topk={num_topk}",
            flush=True,
        )

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    x_e4m3 = per_token_cast_to_fp8(x)
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )

    # Simplified routing for PyTorch distributed
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(
        group_scores, k=num_topk_groups, dim=-1, sorted=False
    ).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[
        1
    ]
    topk_weights = (
        torch.ones((num_tokens, num_topk), dtype=torch.float32, device="cuda") * rank
    )
    topk_weights_pure_rand = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="cuda"
    )

    # Simplified rank indexing for PyTorch distributed
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="cuda")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta - simplified for PyTorch distributed
    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
    token_idx_in_rank = torch.full(
        (num_ranks, num_tokens), -1, dtype=torch.long, device="cuda"
    )
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(
            count, dtype=torch.long, device="cuda"
        )

    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    if local_rank == 0:
        print(f"[layout] Layout computed", flush=True)
        print("", flush=True)
    group.barrier()
    time.sleep(1)

    # Test dispatch with PyTorch distributed communication
    def check_data(check_x, recv_gbl_rank_prefix_sum):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = recv_gbl_rank_prefix_sum[i].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0

    def pytorch_dispatch(x, topk_idx=None, topk_weights=None):
        """Simple PyTorch distributed dispatch implementation"""
        # Calculate which tokens go to which rank
        tokens_per_rank = torch.zeros(num_ranks, dtype=torch.int, device="cuda")
        for i in range(num_ranks):
            tokens_per_rank[i] = (rank_idx == i).sum()

        # Gather all token counts
        all_tokens_per_rank = [
            torch.zeros_like(tokens_per_rank) for _ in range(num_ranks)
        ]
        dist.all_gather(all_tokens_per_rank, tokens_per_rank, group=group)
        all_tokens_per_rank = torch.stack(all_tokens_per_rank)

        # Calculate prefix sums
        prefix_sums = torch.cumsum(all_tokens_per_rank, dim=0)

        # Extract tokens for this rank
        start_idx = prefix_sums[rank] - all_tokens_per_rank[rank]
        end_idx = prefix_sums[rank]

        # Create mask for tokens that belong to this rank
        rank_mask = rank_idx == rank
        rank_tokens = x[rank_mask]

        # Gather all tokens from all ranks
        gathered_tokens = [torch.zeros_like(rank_tokens) for _ in range(num_ranks)]
        dist.all_gather(gathered_tokens, rank_tokens, group=group)

        # Concatenate all gathered tokens
        all_tokens = torch.cat(gathered_tokens, dim=0)

        # Extract tokens for this rank based on prefix sums
        my_tokens = all_tokens[start_idx:end_idx]

        return my_tokens, prefix_sums[rank]

    def pytorch_combine(x, prefix_sums):
        """Simple PyTorch distributed combine implementation"""
        # Scatter tokens back to their original ranks
        tokens_per_rank = torch.zeros(num_ranks, dtype=torch.int, device="cuda")
        for i in range(num_ranks):
            tokens_per_rank[i] = (rank_idx == i).sum()

        # Calculate how many tokens this rank should receive
        my_token_count = tokens_per_rank[rank]

        # Create output tensor
        output = torch.zeros((num_tokens, hidden), dtype=x.dtype, device="cuda")

        # Scatter tokens back
        for i in range(num_ranks):
            if i == rank:
                continue
            start_idx = prefix_sums[i] - tokens_per_rank[i]
            end_idx = prefix_sums[i]
            if start_idx < end_idx:
                # This is a simplified scatter - in practice you'd use all_to_all
                rank_tokens = x[start_idx:end_idx]
                # Find where these tokens should go in the output
                rank_mask = rank_idx == i
                output[rank_mask] = rank_tokens

        return output

    for previous_mode in (False, True):
        for async_mode in (False, True):
            for current_x in (x_pure_rand, x, x_e4m3):
                for with_topk in (False, True):
                    if local_rank == 0:
                        print(
                            f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...',
                            flush=True,
                            end="",
                        )

                    # PyTorch distributed dispatch
                    recv_x, prefix_sums = pytorch_dispatch(current_x)

                    # PyTorch distributed combine
                    combined_x = pytorch_combine(recv_x, prefix_sums)

                    # Checks
                    if current_x is not x_pure_rand:
                        check_data(combined_x, prefix_sums)

                    if local_rank == 0:
                        print(" passed", flush=True)

    if local_rank == 0:
        print("", flush=True)

    output_data = {}

    # Tune PyTorch distributed performance
    best_time, best_results = 1e10, None

    # Test different chunk sizes for PyTorch distributed
    for chunk_size in range(1024, 8193, 1024):

        def pytorch_dispatch_chunked(x):
            # Implement chunked dispatch for better performance
            chunked_x = x.chunk(chunk_size, dim=0)
            results = []
            for chunk in chunked_x:
                result, _ = pytorch_dispatch(chunk)
                results.append(result)
            return torch.cat(results, dim=0), None

        tune_args = {"x": x}
        t = bench(lambda: pytorch_dispatch_chunked(**tune_args))[0]

        if t < best_time:
            best_time, best_results = t, (chunk_size,)

        if local_rank == 0:
            print(
                f"[tuning] Chunk size {chunk_size}: {x.numel() * 2 / 1e9 / t:.2f} GB/s",
                flush=True,
            )

    if local_rank == 0:
        print(
            f"[tuning] Best PyTorch distributed: Chunk size {best_results[0]}: {x.numel() * 2 / 1e9 / best_time:.2f} GB/s",
            flush=True,
        )
        print("", flush=True)
        output_data["pytorch_distributed"] = {"chunk_size": best_results[0]}

    if rank == 0 and local_rank == 0:
        _write_output(args, output_data)


def _write_output(args, output_data):
    text = json.dumps(output_data, indent=4)
    output_path = args.output_path
    print(f"Write to {output_path} with {text}")
    Path(output_path).write_text(text)


def test_loop(local_rank: int, num_local_ranks: int, args):
    num_nodes = args.nnodes
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks, args)

    num_sms = args.num_sms
    torch.manual_seed(rank)

    for i in (num_sms,):
        test_main(
            i,
            local_rank,
            num_local_ranks,
            num_ranks,
            num_nodes,
            rank,
            group,
            args,
        )
        if local_rank == 0:
            print("", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-sms", type=int, default=24)
    parser.add_argument(
        "--output-path", type=str, default="deepep_tuned_pytorch_dist.json"
    )
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=8361)
    args = parser.parse_args()
    print(f"Start system with {args=}")

    num_processes = 8
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
