import os
import torch
import torch.distributed as dist


def main():
    # Set NCCL debug info
    os.environ['NCCL_DEBUG'] = 'INFO'
    
    # Initialize distributed process group
    dist.init_process_group(backend='nccl')
    
    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device
    device = torch.device(f'cuda:{rank % 4}')
    torch.cuda.set_device(device)
    
    # Simple all-reduce test
    tensor = torch.ones(1, device=device, dtype=torch.int32)
    
    dist.all_reduce(tensor)
    
    # Cleanup
    dist.destroy_process_group()

    if rank == 0:
        if tensor == world_size:
            print("SUCCESS!")
        else:
            print(f"FAILURE!, expected {world_size}, got {tensor.item()}")

if __name__ == "__main__":
    main()
