import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        return self.layers(x)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    print(f"Running on rank {rank}")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = SimpleNet().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    batch_size = 32
    input_data = torch.randn(batch_size, 1000).to(rank)
    target = torch.randn(batch_size, 10).to(rank)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

    print(f"Rank {rank}: Starting training")
    for iteration in range(10):
        optimizer.zero_grad()
        output = ddp_model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if rank == 0 and iteration % 2 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item():.6f}")

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs!")
    
    # Print GPU information
    for i in range(world_size):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Print PyTorch and CUDA versions
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    try:
        mp.spawn(train,
                args=(world_size,),
                nprocs=world_size,
                join=True)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()