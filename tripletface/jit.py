"""jit.py

This script Just-In-Time compile the models and save it
"""
import torch
from tripletface.core.model import Encoder

"""model initialization

This part define the model and load his weights.
"""
print("Loading model & weight...\n")
model = Encoder(64).cuda() # embeding_size = 64
weight = torch.load("model.pt")['model']
model.load_state_dict(weight)
print("Model & weight loaded\n")


"""JIT making and save

This part generate a Just-In-Time trace from the weighted model and save it as
scriptmodule.pt
"""
print("JIT making...\n")
input = torch.randn((1, 3, 299, 299)).float()
module = torch.jit.trace(model, input, check_trace=False)
torch.jit.save(module, "scriptmodule.pt")
print("JIT done & saved")
