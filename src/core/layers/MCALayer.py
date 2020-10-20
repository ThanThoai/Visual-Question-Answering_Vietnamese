import torch
from .SelfAtt import SelfAtt
from .SelfGuidedAtt import SelfGuidedAtt

class MCALayers(torch.nn.Module):
    
    def __init__(self, __C):
        super(MCALayers, self).__init__()
        self.enc_list = torch.nn.ModuleList([SelfAtt(__C) for _ in range(__C.LAYER)])
        self.dec_list = torch.nn.ModuleList([SelfGuidedAtt(__C) for _ in range(__C.LAYER)])
        
    
    def forward(self, y, x, y_mask, x_mask):
        
        for enc in self.enc_list:
            
            y = enc(y, y_mask, x)
            
        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask)
            
        return y, x