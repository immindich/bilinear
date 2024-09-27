import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
from interpolated_ffn import ModelWithBilinearLayer, save_layer
import signal
import os
from torch.utils.tensorboard import SummaryWriter

device = 'cuda'
dtype = torch.bfloat16

data = np.memmap('data_train.bin', dtype=np.uint32, mode='r')

def sample_batch(size, seq_len):
    indices = torch.randint(len(data) - seq_len - 1, (size,))
    xs = torch.stack([torch.from_numpy(data[i:i+seq_len].astype(np.int64)) for i in indices])
    return xs.to(device)

model_name = "gemma-2-2b"
model_pretrained = HookedTransformer.from_pretrained_no_processing(model_name, device = device, dtype=dtype)

layer = 18
model_bilinear = ModelWithBilinearLayer(model_pretrained, layer)

pause_training = False
stop_training = False

def run_name(layer, step):
    return f"bilinear-layer-{layer}-step-{step}"

def pause_handler(number, frame):
    global pause_training
    global stop_training
    pause_training = True
    stop_training = number == signal.SIGUSR2

def train(model, steps, minibatch_size, batch_size, seq_len, lr=1e-5):
    global pause_training
    global stop_training

    params = model.ffn.parameters()
    opt = torch.optim.AdamW(params, lr=lr)

    writer = SummaryWriter(os.path.join("runs", f"bilinear-layer-{layer}"))

    for i in range(steps):
        opt.zero_grad()
        batch_loss = 0.0
        for j in range(batch_size):
            minibatch = sample_batch(minibatch_size, seq_len)
            with torch.no_grad():
                res = model_pretrained(minibatch, stop_at_layer=layer)
                logits_correct = model_pretrained(res, start_at_layer=layer)
            logits = model_bilinear.run_from_modified_layer(res)
            loss = F.mse_loss(logits, logits_correct) / batch_size
            loss.backward()
            batch_loss += loss.item()
        opt.step()

        print(f"Step {i}: {batch_loss}")
        writer.add_scalar("training loss/step", batch_loss, i)

        if pause_training:
            print("Signal received, saving checkpoint")
            save_layer(model, run_name(layer, i) + ".pt")
            if stop_training:
                return
            pause_training = False
            stop_training = False

    save_layer(model, run_name(layer, steps) + ".pt")

steps = 10000
minibatch_size = 2
batch_size = 2
seq_len = 1024
lr = 1e-5

signal.signal(signal.SIGUSR1, pause_handler)
signal.signal(signal.SIGUSR2, pause_handler)

train(model_bilinear, steps, minibatch_size, batch_size, seq_len, lr=lr)
