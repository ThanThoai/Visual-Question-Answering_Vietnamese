import torch
from .FFN import FNN
from .MultiHeadAtt import MultiHeadAtt
from .LayerNorm import LayerNorm


class SelfAtt(torch.nn.Module):
    
    def __init__(self, __C):
        super(SelfAtt, self).__init__()
        
        self.mhatt = MultiHeadAtt(__C)
        self.ffn   = FNN(__C)
        
        self.dropout_1 = torch.nn.Dropout(__C.DROPOUT_R)
        self.norm_1    = LayerNorm(__C.HIDDEN_SIZE)
        
        self.dropout_2 = torch.nn.Dropout(__C.DROPOUT_R)
        self.norm_2    = LayerNorm(__C.HIDDEN_SIZE)
    
    def forward(self, y, y_mask):
        
        y = self.norm_1(y + self.dropout_1(self.mhatt(y, y, y, y_mask)))
        y = self.norm_2(y + self.dropout_2(self.ffn(y)))
        return y   