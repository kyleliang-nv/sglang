# MODIFIED FROM tuning_deepep.py for non-RDMA communication

"""
Example usage for non-RDMA setups:
python tuning_deepep_non_rdma.py --nnodes 1 --node-rank 0 --master-addr 127.0.0.1
Then check `deepep_tuned_non_rdma.json`
"""

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

# noinspection PyUnresolvedReferences
import deep_ep
import torch
import torch.distributed as dist
from deepep_utils import (
    bench,
    calc_diff,
    create_grouped_scores,
    init_dist,
    inplace_unique,
    per_token_cast_back,
    per_token_cast_to_fp8,
)


def test_main(
    num_sms: int,
    local_rank: int,
    num_local_ranks: int,
    num_ranks: int,
    num_nodes: int,
    rank: int,
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
    args,
):
    # Settings - simplified for single-node or NVL-only communication
    num_tokens, hidden, num_topk_groups, num_topk, num_experts = (
        4096,
        7168,
        min(num_nodes, 4),
        8,
        (256 // num_ranks) * num_ranks,
    )
    assert num_experts % num_ranks == 0 and num_local_ranks <= 8
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

    # Simplified routing for NVL-only communication
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

    # Simplified rank indexing for NVL-only
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # For NVL-only, treat all communication as within-node
    nvl_rank_idx = rank_idx.clone()
    nvl_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(nvl_rank_idx, num_ranks)

    # NVL dispatch counts (no RDMA)
    nvl_idx = topk_idx // (num_experts // num_ranks)
    nvl_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(nvl_idx, num_ranks)
    num_nvl_token_sent = nvl_idx.ne(-1).sum().item()

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="cuda")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta - simplified for NVL-only
    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
    num_tokens_per_nvl_rank = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
    token_idx_in_rank = torch.full(
        (num_ranks, num_tokens), -1, dtype=torch.long, device="cuda"
    )
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        num_tokens_per_nvl_rank[i] = (nvl_rank_idx == i).sum()
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

    # Test dispatch layout
    # In low latency mode, get_dispatch_layout might not be available or might return different values
    try:
        (
            ref_num_tokens_per_rank,
            ref_num_tokens_per_nvl_rank,
            ref_num_tokens_per_expert,
            ref_is_token_in_rank,
            _,
        ) = buffer.get_dispatch_layout(topk_idx, num_experts)
        assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
        assert torch.allclose(ref_num_tokens_per_nvl_rank, num_tokens_per_nvl_rank)
        assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
        assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
        t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    except (AttributeError, TypeError, RuntimeError):
        # In low latency mode, skip layout validation and timing
        if local_rank == 0:
            print("[layout] Skipping layout validation in low latency mode", flush=True)
        t = 0.0
    if local_rank == 0:
        print(f"[layout] Kernel performance: {t * 1000:.3f} ms", flush=True)
        print("", flush=True)
    group.barrier()
    time.sleep(1)

    # Config - NVL-only
    nvl_buffer_size = 512  # Simplified buffer size
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size, 16, 0)  # RDMA buffer size = 0

    # Test dispatch with NVL-only communication
    def check_data(check_x, recv_gbl_rank_prefix_sum):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = recv_gbl_rank_prefix_sum[i].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0

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
                    dispatch_args = {
                        "x": current_x,
                        "num_tokens_per_rank": num_tokens_per_rank,
                        "num_tokens_per_rdma_rank": num_tokens_per_nvl_rank,  # Use NVL rank instead
                        "is_token_in_rank": is_token_in_rank,
                        "num_tokens_per_expert": num_tokens_per_expert,
                        "config": config,
                        "async_finish": async_mode,
                    }
                    if with_topk:
                        dispatch_args.update(
                            {
                                "topk_idx": topk_idx,
                                "topk_weights": (
                                    topk_weights_pure_rand
                                    if current_x is x_pure_rand
                                    else topk_weights
                                ),
                            }
                        )
                    if previous_mode:
                        dispatch_args.update({"previous_event": buffer.capture()})
                    (
                        recv_x,
                        recv_topk_idx,
                        recv_topk_weights,
                        recv_num_tokens_per_expert_list,
                        handle,
                        event,
                    ) = buffer.dispatch(**dispatch_args)
                    event.current_stream_wait() if async_mode else ()
                    recv_x = (
                        per_token_cast_back(*recv_x)
                        if isinstance(recv_x, tuple)
                        else recv_x
                    )

                    # Checks
                    recv_gbl_rank_prefix_sum = handle[-4]
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(
                        0
                    ), f"{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}"
                    assert (
                        gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
                        == recv_num_tokens_per_expert_list
                    )
                    if current_x is not x_pure_rand:
                        check_data(recv_x, recv_gbl_rank_prefix_sum)
                    if with_topk:
                        # Check `topk_idx`
                        assert (
                            recv_topk_idx.eq(-1)
                            | (
                                (recv_topk_idx >= 0)
                                & (recv_topk_idx < (num_experts // num_ranks))
                            )
                        ).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count

                        # Check `topk_weights`
                        if current_x is not x_pure_rand:
                            recv_topk_weights[recv_topk_idx.eq(-1)] = (
                                recv_topk_weights.amax(dim=1, keepdim=True).expand_as(
                                    recv_topk_weights
                                )[recv_topk_idx.eq(-1)]
                            )
                            check_data(recv_topk_weights, recv_gbl_rank_prefix_sum)

                    # Test cached dispatch (must without top-k staffs)
                    if not with_topk:
                        dispatch_args = {
                            "x": current_x,
                            "handle": handle,
                            "config": config,
                            "async_finish": async_mode,
                        }
                        if previous_mode:
                            dispatch_args.update({"previous_event": buffer.capture()})
                        recv_x, _, _, _, event = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()
                        recv_x = (
                            per_token_cast_back(*recv_x)
                            if isinstance(recv_x, tuple)
                            else recv_x
                        )
                        if current_x is not x_pure_rand:
                            check_data(recv_x, recv_gbl_rank_prefix_sum)

                    # Test combine
                    combine_args = {
                        "x": recv_x,
                        "handle": handle,
                        "config": config,
                        "async_finish": async_mode,
                    }
                    if with_topk:
                        combine_args.update({"topk_weights": recv_topk_weights})
                    if previous_mode:
                        combine_args.update({"previous_event": buffer.capture()})
                    combined_x, combined_topk_weights, event = buffer.combine(
                        **combine_args
                    )
                    event.current_stream_wait() if async_mode else ()
                    check_x = combined_x.float() / is_token_in_rank.sum(
                        dim=1
                    ).unsqueeze(1)
                    ref_x = x_pure_rand if current_x is x_pure_rand else x
                    assert calc_diff(check_x, ref_x) < 5e-6
                    if with_topk:
                        check_topk_weights = (
                            combined_topk_weights
                            if (current_x is x_pure_rand)
                            else (
                                combined_topk_weights
                                / is_token_in_rank.sum(dim=1).unsqueeze(1)
                            )
                        )
                        ref_topk_weights = (
                            topk_weights_pure_rand
                            if current_x is x_pure_rand
                            else topk_weights
                        )
                        assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                    # For later tuning - NVL-only metrics
                    dispatch_nvl_send_bytes = num_nvl_token_sent * hidden * 2
                    dispatch_nvl_recv_bytes = recv_x.numel() * 2
                    combine_nvl_send_bytes = dispatch_nvl_recv_bytes
                    combine_nvl_recv_bytes = dispatch_nvl_send_bytes

                    if local_rank == 0:
                        print(" passed", flush=True)
    if local_rank == 0:
        print("", flush=True)

    output_data = {}

    # Tune dispatch performance - NVL-only
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in (x_e4m3, x):
        best_time, best_results = 1e10, None
        nvl_send_bytes = (
            (dispatch_nvl_send_bytes * fp8_factor)
            if isinstance(current_x, tuple)
            else dispatch_nvl_send_bytes
        )
        nvl_recv_bytes = (
            (dispatch_nvl_recv_bytes * fp8_factor)
            if isinstance(current_x, tuple)
            else dispatch_nvl_recv_bytes
        )

        # Focus only on NVL chunk sizes
        for nvl_chunk_size in range(4, 65, 4):  # Wider range for NVL-only
            config_kwargs = {
                "num_sms": num_sms,
                "num_max_nvl_chunked_send_tokens": nvl_chunk_size,
                "num_max_nvl_chunked_recv_tokens": nvl_buffer_size,
                "num_max_rdma_chunked_send_tokens": 0,  # No RDMA
                "num_max_rdma_chunked_recv_tokens": 0,  # No RDMA
            }
            config = deep_ep.Config(**config_kwargs)
            tune_args = {"x": current_x, "handle": handle, "config": config}
            t = bench(lambda: buffer.dispatch(**tune_args))[0]
            if t < best_time:
                best_time, best_results = t, (
                    num_sms,
                    nvl_chunk_size,
                    0,  # No RDMA chunk size
                    config_kwargs,
                )
            if local_rank == 0:
                print(
                    f"[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}: {nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL send), {nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL recv) ",
                    flush=True,
                )
        if local_rank == 0:
            print(
                f'[tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): SMs {best_results[0]}, NVL chunk {best_results[1]}: {nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL send), {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL recv)',
                flush=True,
            )
            print("", flush=True)
            is_fp8 = isinstance(current_x, tuple)
            if is_fp8:
                output_data["normal_dispatch"] = deepcopy(best_results[3])

        if isinstance(current_x, tuple):
            # Gather FP8 the best config from rank 0
            best_dispatch_results = torch.tensor(
                [best_results[0], best_results[1], 0],  # No RDMA chunk size
                dtype=torch.int32,
                device="cuda",
            )
            all_best_fp8_results_list = [
                torch.zeros_like(best_dispatch_results)
                for _ in range(torch.distributed.get_world_size())
            ]
            dist.all_gather(
                all_best_fp8_results_list, best_dispatch_results, group=group
            )
            best_dispatch_results = all_best_fp8_results_list[0].tolist()

    dispatch_config = deep_ep.Config(
        best_dispatch_results[0],
        best_dispatch_results[1],
        nvl_buffer_size,
        0,  # No RDMA chunk size
        0,  # No RDMA buffer size
    )

    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "num_tokens_per_rdma_rank": num_tokens_per_nvl_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": dispatch_config if dispatch_config is not None else config,
    }
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance - NVL-only
    best_time, best_results = 1e10, None
    for nvl_chunk_size in range(1, 9, 1):  # Wider range for NVL-only
        config_kwargs = {
            "num_sms": num_sms,
            "num_max_nvl_chunked_send_tokens": nvl_chunk_size,
            "num_max_nvl_chunked_recv_tokens": nvl_buffer_size,
            "num_max_rdma_chunked_send_tokens": 0,  # No RDMA
            "num_max_rdma_chunked_recv_tokens": 0,  # No RDMA
        }
        config = deep_ep.Config(**config_kwargs)
        tune_args = {"x": recv_x, "handle": handle, "config": config}
        t = bench(lambda: buffer.combine(**tune_args))[0]
        if local_rank == 0:
            print(
                f"[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}: {combine_nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL recv), {combine_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL send) ",
                flush=True,
            )
            if t < best_time:
                best_time, best_results = t, (
                    num_sms,
                    nvl_chunk_size,
                    0,  # No RDMA chunk size
                    config_kwargs,
                )

    if local_rank == 0:
        print(
            f"[tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}: {combine_nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL recv), {combine_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL send)",
            flush=True,
        )
        print("", flush=True)
        output_data["normal_combine"] = deepcopy(best_results[3])

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
    num_experts = (256 // num_ranks) * num_ranks
    num_qps_per_rank_regular = num_sms // 2  # For regular mode
    num_qps_per_rank_low_latency = num_experts // num_ranks  # For low latency mode

    # Try regular buffer initialization first, fall back to low latency mode if needed
    try:
        buffer = deep_ep.Buffer(
            group,
            int(1e9),
            int(1e9),
            low_latency_mode=False,
            num_qps_per_rank=num_qps_per_rank_regular,
        )
    except RuntimeError as e:
        if "num_ranks > NUM_MAX_NVL_PEERS" in str(e):
            # Fall back to low latency mode
            num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
                4096, 7168, num_ranks, num_experts
            )
            buffer = deep_ep.Buffer(
                group,
                num_rdma_bytes=num_rdma_bytes,
                low_latency_mode=True,
                num_qps_per_rank=num_qps_per_rank_low_latency,
            )
        else:
            raise
    assert num_local_ranks <= 8 and num_ranks >= num_local_ranks
    torch.manual_seed(rank)

    try:
        for i in (num_sms,):
            test_main(
                i,
                local_rank,
                num_local_ranks,
                num_ranks,
                num_nodes,
                rank,
                buffer,
                group,
                args,
            )
            if local_rank == 0:
                print("", flush=True)
    finally:
        # Clean up distributed process group
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-sms", type=int, default=24)
    parser.add_argument("--output-path", type=str, default="deepep_tuned_non_rdma.json")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=8361)
    args = parser.parse_args()
    print(f"Start system with {args=}")

    # Dynamically determine number of processes based on available CUDA devices
    num_cuda_devices = torch.cuda.device_count()
    num_processes = min(
        8, num_cuda_devices
    )  # Use at most 8 processes, but not more than available devices

    if num_cuda_devices < 8:
        print(
            f"Warning: Only {num_cuda_devices} CUDA devices available, using {num_processes} processes instead of 8"
        )
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
