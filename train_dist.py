import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
from interpolated_ffn import ModelWithBilinearLayer, save_layer
import os
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def sample_batch(size, seq_len, data):
    indices = torch.randint(len(data) - seq_len - 1, (size,))
    xs = torch.stack([torch.from_numpy(data[i:i+seq_len].astype(np.int64)) for i in indices])
    return xs

def run_name(layer, step):
    return f"bilinear-layer-{layer}-step-{step}"

pause_training = False
stop_training = False

def train(rank, world_size, model_name, layer, steps, minibatch_size, batch_size, seq_len, lr=1e-5):
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    dtype = torch.bfloat16

    data = np.memmap('data_train.bin', dtype=np.uint32, mode='r')

    model_pretrained = HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype=dtype)
    model_bilinear = ModelWithBilinearLayer(model_pretrained, layer).to(device)
    model_bilinear.ffn = DDP(model_bilinear.ffn, device_ids=[rank])

    params = model_bilinear.ffn.parameters()
    opt = torch.optim.AdamW(params, lr=lr)

    if rank == 0:
        writer = SummaryWriter(os.path.join("runs", f"bilinear-layer-{layer}"))

    for i in range(steps):
        opt.zero_grad()
        batch_loss = 0.0
        for j in range(batch_size):
            minibatch = sample_batch(minibatch_size, seq_len, data).to(device)
            with torch.no_grad():
                logits_correct = model_pretrained(minibatch)
            logits = model_bilinear(minibatch)
            loss = F.mse_loss(logits, logits_correct) / batch_size
            loss.backward()
            batch_loss += loss.item()
        opt.step()

        if rank == 0:
            print(f"Step {i}: {batch_loss}")
            writer.add_scalar("training loss/step", batch_loss, i)

    if rank == 0:
        save_layer(model_bilinear.module, run_name(layer, steps) + ".pt")
    
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    model_name = "gemma-2-2b"
    layer = 18
    steps = 10000
    minibatch_size = 1
    batch_size = 2
    seq_len = 2
    lr = 1e-5

    print("World size:", world_size)
    assert batch_size % world_size == 0
    batch_size //= world_size

    mp.spawn(
        train,
        args=(world_size, model_name, layer, steps, minibatch_size, batch_size, seq_len, lr),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()