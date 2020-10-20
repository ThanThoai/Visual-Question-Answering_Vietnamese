import torch

def make_mask(feature):
    
    return (torch.sum(torch.abs(feature), dim = -1) == 0).unsqueeze(1).unsqueeze(2)