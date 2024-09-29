import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
from interpolated_ffn import ModelWithBilinearLayer, save_layer
import signal
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

device = 'cuda'
dtype = torch.bfloat16

data = np.memmap('data_train.bin', dtype=np.uint32, mode='r')

def sample_batch(size, seq_len):
    indices = torch.randint(len(data) - seq_len - 1, (size,))
    xs = torch.stack([torch.from_numpy(data[i:i+seq_len].astype(np.int64)) for i in indices])
    ys = torch.stack([torch.from_numpy(data[i+1:i+seq_len+1].astype(np.int64)) for i in indices])
    return xs.to(device), ys.to(device)

model_name = "gemma-2-2b"
model_pretrained = HookedTransformer.from_pretrained_no_processing(model_name, device = device, dtype=dtype)

pause_training = False
stop_training = False

def run_name(layer, step, name):
    return f"layer-{layer}-step-{step}-{name}"

def pause_handler(number, frame):
    global pause_training
    global stop_training
    pause_training = True
    stop_training = number == signal.SIGUSR2

def logit_reconstruction(model, res, next_tokens):
    with torch.no_grad():
        logits_correct = model_pretrained(res, start_at_layer=layer)
    logits = model.run_from_modified_layer(res)
    return F.mse_loss(logits, logits_correct)

def output_reconstruction(model, res, next_tokens):
    with torch.no_grad():
        correct_output = model_pretrained(res, start_at_layer=layer, stop_at_layer=layer+1)
    output = model.newlayer(res)
    return F.mse_loss(output, correct_output)

def next_token_ce(model, res, next_tokens):
    logits = model.run_from_modified_layer(res)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), next_tokens.view(-1))

def train(model, name, steps, minibatch_size, batch_size, seq_len, lr=1e-5, loss_fn=logit_reconstruction):
    global pause_training
    global stop_training

    params = model.ffn.parameters()
    opt = torch.optim.AdamW(params, lr=lr)

    writer = SummaryWriter(os.path.join("runs", f"layer-{layer}-{name}"))

    for i in tqdm(range(steps)):
        opt.zero_grad()
        batch_loss = 0.0
        for j in range(batch_size):
            minibatch, next_tokens = sample_batch(minibatch_size, seq_len)
            with torch.no_grad():
                res = model_pretrained(minibatch, stop_at_layer=layer)
            loss = loss_fn(model, res, next_tokens) / batch_size
            loss.backward()
            batch_loss += loss.item()
        opt.step()

        writer.add_scalar("training loss/step", batch_loss, i)

        if pause_training:
            print("Signal received, saving checkpoint")
            save_layer(model, run_name(layer, i, name) + ".pt")
            if stop_training:
                return
            pause_training = False
            stop_training = False

        if i != 0 and i % 1000 == 0:
            save_layer(model, run_name(layer, i, name) + ".pt")

    save_layer(model, run_name(layer, steps, name) + ".pt")
    writer.flush()

steps = 20_000
minibatch_size = 5
batch_size = 2
seq_len = 1024
lr = 3e-6

signal.signal(signal.SIGUSR1, pause_handler)
signal.signal(signal.SIGUSR2, pause_handler)

layer = 18

model_bilinear_logit_reconstruction = ModelWithBilinearLayer(model_pretrained, layer)
train(model_bilinear_logit_reconstruction, "logit-mse", steps, minibatch_size, batch_size, seq_len, lr=lr, loss_fn=logit_reconstruction)

model_bilinear_output_reconstruction = ModelWithBilinearLayer(model_pretrained, layer)
train(model_bilinear_output_reconstruction, "output-mse", steps, minibatch_size, batch_size, seq_len, lr=lr, loss_fn=output_reconstruction)

model_bilinear_ce = ModelWithBilinearLayer(model_pretrained, layer)
train(model_bilinear_ce, "ce", steps, minibatch_size, batch_size, seq_len, lr=lr, loss_fn=next_token_ce)


