import random
import torch
import torch.distributed as dist
from functools import partial

import time
import numpy as np
import copy

import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back


def test_main(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
              rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer, seed: int = 0):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks
    num_rdma_ranks = num_ranks / 8

    # NOTES: the integers greater than 256 exceeds the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device='cuda').to(torch.bfloat16).view(-1, 1)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()
    # print("topk_weights: ", topk_weights)

    # Randomly mask some positions
    for i in range(10):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += num_bf16_bytes * num_selections

    # Check dispatch correctness
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
    cache.zero_()
    torch.cuda.synchronize()
    group.barrier()
    do_check = True
    hash_value, num_times = 0, 0
    all_times = 1000
    warp_up_time = 0
    dispatch_times = []
    combine_times = []
    for return_recv_hook in (False, ):
        for dispatch_use_fp8 in (True, ): # True False
            print("dispatch_use_fp8: ", dispatch_use_fp8)
            ############dispatch#############
            num_times += 1
            # for i in range((num_times % 2) + 1):
            # if (rank == 0 or rank == 8):
            #     print("-------------------input-------------------")
            #     print("x: ", x.dtype, x.shape, x)
            #     print(f"rank: {rank}, topk_idx: {topk_idx.dtype, topk_idx.shape, topk_idx}, topk_weights: {topk_weights.dtype, topk_weights.shape, topk_weights}", flush=True)
                # tmp_topk = topk_idx == -1
                # print(f"rank {rank}, total token: {(tmp_topk.sum(axis=1) == 8).sum()}", flush=True)
                # print("num_tokens: ", num_tokens)
                # print("num_experts: ", num_experts)
                # print("-------------------end-------------------")
            for i in range(all_times):
                ############dispatch#############
                if rank == 0 or rank == 8:
                    print("dispatch i: ", i, flush=True)
                packed_recv_x, packed_recv_count, rdma_send_flags, handle, event, hook = \
                    buffer.low_latency_dispatch_two_stage(
                        x, 
                        topk_idx, 
                        topk_weights,
                        # x_copy,
                        # topk_idx_copy,
                        # topk_weights_copy,
                        num_tokens, 
                        num_experts, 
                        use_fp8=dispatch_use_fp8,
                        async_finish=not return_recv_hook, 
                        return_recv_hook=return_recv_hook)
                packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous()) if dispatch_use_fp8 else packed_recv_x
                ############combine#############
                if rank == 0 or rank == 8:
                    print("combine i: ", i, flush=True)
                out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                if dispatch_use_fp8:
                    simulated_gemm_x = per_token_cast_back(packed_recv_x[0].view(-1, hidden), packed_recv_x[1].view(-1, hidden // 128)).view(packed_recv_x[0].shape)
                else:
                    simulated_gemm_x = packed_recv_x.clone()
                combined_x, event, hook = buffer.low_latency_combine_two_stage(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                                        async_finish=not return_recv_hook, dispatch_use_fp8=dispatch_use_fp8,
                                                                        return_recv_hook=return_recv_hook, out=out)
                torch.cuda.synchronize()
                dist.barrier()
                if i > 0:
                    if (rank == 0 or rank == 8):
                        print(f"rank: {rank}, is_same: {torch.allclose(old_combined_x, combined_x)}, max_diff: {torch.max(torch.abs(old_combined_x - combined_x))}", flush=True)
                if i == 0:
                    old_combined_x = copy.deepcopy(combined_x)
                if (rank == 0 or rank == 8):
                    print(f"rank: {rank}, combined_x: {combined_x.dtype, combined_x.shape, combined_x}", flush=True)
            
            def test_func(return_recv_hook: bool):
                packed_recv_x, packed_recv_count, rdma_send_flags, handle, event, hook = \
                    buffer.low_latency_dispatch_two_stage(
                        x, 
                        topk_idx, 
                        topk_weights,
                        num_tokens, 
                        num_experts, 
                        use_fp8=dispatch_use_fp8,
                        async_finish=not return_recv_hook, 
                        return_recv_hook=return_recv_hook)
                combined_x, event, hook = buffer.low_latency_combine_two_stage(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                                        async_finish=not return_recv_hook, dispatch_use_fp8=dispatch_use_fp8,
                                                                        return_recv_hook=return_recv_hook, out=out)
            dispatch + combine
            avg_t, min_t, max_t = bench(partial(test_func, return_recv_hook=False), num_warmups=200, num_tests=10000)  
            print(f'[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
                  f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us', flush=True)   
            # dispatch / combine
            dispatch_t, combine_t = bench_kineto(partial(test_func, return_recv_hook=False),
                                             kernel_names=('dispatch', 'combine'), barrier_comm_profiling=True,
                                             suppress_kineto_output=True, num_tests=10000)
            print(f'[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | '
                  f'Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us', flush=True)
            print(f'[rank {rank}] Dispatch + combine bandwidth111: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / (dispatch_t + combine_t):.2f} GB/s, avg_t={(dispatch_t + combine_t) * 1e6:.2f} us', flush=True)   
    return hash_value


# noinspection PyUnboundLocalVariable
def test_loop(local_rank: int, num_local_ranks: int):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden, num_topk, num_experts = 4096, 7168, 8, 64
    num_rdma_ranks = num_ranks / 8
    num_local_experts = num_experts / num_ranks
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint_two_stage(num_tokens, hidden, num_ranks, num_experts, num_topk)
    use_fp8 = True
    # nvl_recv_x: num_local_experts * dp_num * num_max_token_per_dp * hidden_size
    # nvl_recv_count: num_local_experts * dp_num
    num_nvl_bytes = deep_ep.Buffer.get_low_latency_nvl_size_hint_two_stage(num_tokens, hidden, num_ranks, num_experts, num_topk, use_fp8)
    if local_rank == 0:
        print(f'Allocating rdma buffer size: {num_rdma_bytes / 1e6} MB, nvl buffer size: {num_nvl_bytes / 1e6} MB...', flush=True)
    buffer = deep_ep.Buffer(group, 
                            num_nvl_bytes=num_nvl_bytes,
                            num_rdma_bytes=num_rdma_bytes, 
                            low_latency_mode=True,
                            num_qps_per_rank=num_rdma_ranks)
    test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer, seed=1)

    # do_pressure_test = False
    # for seed in range(int(1e9) if do_pressure_test else 0):
    #     if local_rank == 0:
    #         print(f'Testing with seed {seed} ...', flush=True)
    #     ref_hash = test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer, seed=seed)
        # for i in range(20):
        #     assert test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer, seed=seed) == ref_hash, f'Error: seed={seed}'


if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    num_processes = 8
    torch.multiprocessing.spawn(test_loop, args=(num_processes,), nprocs=num_processes)