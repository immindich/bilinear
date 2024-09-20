import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer
from interpolated_ffn import ModelWithBilinearLayer, save_layer

device = 'cuda'
dtype = torch.bfloat16

data = np.memmap('data_train.bin', dtype=np.uint32, mode='r')

def sample_batch(size, seq_len):
    indices = torch.randint(len(data) - seq_len - 1, (size,))
    xs = torch.stack([torch.from_numpy(data[i:i+seq_len].astype(np.int64)) for i in indices])
    return xs.to(device)

model_name = "gemma-2-2b"
model = HookedTransformer.from_pretrained_no_processing(model_name, device = device, dtype=dtype)

layer = 18
model_bilinear = ModelWithBilinearLayer(model, layer)

steps = 1000
minibatch_size = 20
batch_size = 2
seq_len = 1024

params = model_bilinear.ffn.parameters()
opt = torch.optim.AdamW(params, lr=1e-5)

for i in range(steps):
    opt.zero_grad()
    batch_loss = 0.0
    for j in range(batch_size):
        minibatch = sample_batch(minibatch_size, seq_len)
        with torch.no_grad():
            logits_correct = model(minibatch)
        logits = model_bilinear(minibatch)
        loss = F.mse_loss(logits, logits_correct) / batch_size
        loss.backward()
        batch_loss += loss.item()
    opt.step()
    print(f"Step {i}: {batch_loss}")

save_layer(model_bilinear, f"bilinear_layer_{layer}.pt")