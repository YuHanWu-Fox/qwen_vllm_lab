"""Check NIXL EP dispatch, combine, and graph replay with real cross-rank routes."""
import argparse
from datetime import timedelta
import faulthandler
import os

import torch
import torch.distributed as dist
from nixl_ep import Buffer, topk_idx_t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    faulthandler.dump_traceback_later(15, repeat=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=35))
    print(f"rank {rank} Gloo store ready", flush=True)
    world_size = dist.get_world_size()
    hidden = 2048
    tokens_per_rank = 32
    local_experts = args.experts // world_size
    store = dist.PrefixStore("nixl_ep_probe", dist.group.WORLD.get_group_store())
    buffer = Buffer(rank=rank, explicitly_destroy=True, tcp_store_group=store, timeout_ms=8000)
    try:
        print(f"rank {rank} Buffer object ready", flush=True)
        bytes_per_rank = buffer.get_rdma_size_hint(
            tokens_per_rank, hidden, world_size, args.experts,
        )
        buffer.update_memory_buffers(
            num_ranks=world_size,
            num_experts_per_rank=local_experts,
            num_rdma_bytes=bytes_per_rank,
        )
        print(f"rank {rank} buffer memory ready", flush=True)
        buffer.connect_ranks(list(range(world_size)))
        print(
            f"rank {rank} connected experts={args.experts} topk={args.topk} "
            f"max_tokens={tokens_per_rank} bytes={bytes_per_rank}",
            flush=True,
        )

        activations = (
            torch.arange(16 * hidden, device="cuda").reshape(16, hidden)
            .remainder(11).add_(rank).to(torch.bfloat16)
        )
        expert_offset = torch.arange(args.topk, dtype=topk_idx_t, device="cuda")
        routes = (
            expert_offset // world_size + expert_offset % world_size * local_experts
        ).expand(activations.shape[0], -1).contiguous()
        scores = torch.full(routes.shape, 1 / args.topk, device="cuda", dtype=torch.float32)
        output = torch.empty_like(activations)

        def round_trip():
            rows, counts, handle, _, receive = buffer.dispatch(
                activations, routes, tokens_per_rank, args.experts,
                use_fp8=False, return_recv_hook=True,
            )
            receive()
            _, _, receive = buffer.combine(
                rows, routes, scores, handle, return_recv_hook=True, out=output,
            )
            receive()
            return counts

        for _ in range(2):
            counts = round_trip()
            assert int(counts.sum().item()) == activations.shape[0] * args.topk
            torch.testing.assert_close(output, activations, atol=0, rtol=0)
        print(f"rank {rank} eager cross-rank round trip exact", flush=True)

        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            round_trip()
        for increment in (1, 2):
            activations.add_(increment)
            dist.barrier()
            graph.replay()
            torch.testing.assert_close(output, activations, atol=0, rtol=0)
        print(f"rank {rank} two FULL graph replays exact", flush=True)
    finally:
        torch.cuda.synchronize()
        buffer.destroy()
        dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
