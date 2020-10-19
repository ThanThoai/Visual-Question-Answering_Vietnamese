import torch


class LayerNorm(torch.nn.Module):
    
    def __init__(self, size , eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        
        self.alpha = torch.nn.Parameter(torch.ones(size))
        self.beta  = torch.nn.Parameter(torch.zeros(size))
        
    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std  = x.std(-1, keepdim = True)
        
        return self.alpha * (x - mean) / (std + self.eps) + self.beta