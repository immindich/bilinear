import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class BilinearFFN(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_hidden)
        self.W2 = nn.Linear(d_model, d_hidden)
        self.V = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.V(self.W1(x) * self.W2(x))
    
# Interpolates between GEGLU and bilinear feedforward
class InterpolatedFFN(nn.Module):
    def __init__(self, cfg, interpolation=0.0):
        super().__init__()
        self.cfg = cfg
        self.interpolation = interpolation
        self.W1 = nn.Parameter(torch.empty(cfg.d_model, cfg.d_mlp))
        self.W2 = nn.Parameter(torch.empty(cfg.d_model, cfg.d_mlp))
        self.V = nn.Parameter(torch.empty(cfg.d_mlp, cfg.d_model))

    def interpolate_gelu(self, x):
        return (1 - self.interpolation) * F.gelu(x) + self.interpolation * x

    def forward(self, x):
        a = x @ self.W1
        b = x @ self.W2
        c = (self.interpolate_gelu(a) * b)
        return c @ self.V
    
class ModelWithInterpolatedLayer(nn.Module):
    def __init__(self, model, layer):
        super().__init__()
        self.model = model
        self.layer_idx = layer
        self.ffn = InterpolatedFFN(model.cfg).to(device=model.cfg.device, dtype=model.cfg.dtype)
        self.ffn.W1.data.copy_(model.blocks[layer].mlp.W_gate)
        self.ffn.W2.data.copy_(model.blocks[layer].mlp.W_in)
        self.ffn.V.data.copy_(model.blocks[layer].mlp.W_out)
        
        self.newlayer = copy.deepcopy(model.blocks[layer])
        self.newlayer.mlp = self.ffn

    def forward(self, tokens):
        with torch.no_grad():
            input = self.model(tokens, stop_at_layer=self.layer_idx)
        output = self.newlayer(input)
        logits = self.model(output, start_at_layer=self.layer_idx+1)
        return logits

    
    def set_interpolation(self, interpolation):
        self.ffn.interpolation = interpolation