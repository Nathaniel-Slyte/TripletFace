import torch
from tripletface.core.model import Encoder

print("Loading model & weight...\n")
model = Encoder(64)
weight = torch.load("model.pt")['model']
model.load_state_dict(weight)
print("Model & weight loaded\n")

input = torch.randn((1, 3, 299, 299)).float()
module = torch.jit.trace(model, input, check_trace=False)
torch.jit.save(module, "scriptmodule.pt")
