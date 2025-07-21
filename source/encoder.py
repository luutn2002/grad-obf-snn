import torch
from spikingjelly.activation_based import encoding
from torch import nn

class Encoder(nn.Module):
    def __init__(self,
                 type_str: str = "rate", 
                 T=10):
        super().__init__()
        self.T = T
        self.encoder, self.mapping_func = self.encoder_getter(type_str)
        self.type_str = type_str

    def encoder_getter(self, type_str: str = "rate"):
        match type_str:
            case "rate": 
                return encoding.PoissonEncoder(), lambda x : x/255.
            case "ttfs": 
                return encoding.LatencyEncoder(T=self.T), lambda x : x/255.
            case "phase": 
                return encoding.WeightedPhaseEncoder(K=8), lambda x : x/256.
            case _:
                raise Exception("Encoder type does not exist.")
        
    def forward(self, x:torch.Tensor):
        x = self.mapping_func(x)
        x = x.clone().detach().to(x.device)
        return torch.stack([self.encoder(x) for _ in range(self.T)])

    def reset(self):
        if self.type_str == "rate": pass
        else: self.encoder.reset()