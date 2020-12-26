import torch
from .FFN import FNN
from .MultiHeadAtt import MultiHeadAtt
from .LayerNorm import LayerNorm


class SelfGuidedAtt(torch.nn.Module):
    
    def __init__(self, __C):
        super(SelfGuidedAtt, self).__init__()
        
        self.mhatt_1 = MultiHeadAtt(__C)
        self.mhatt_2 = MultiHeadAtt(__C)
        
        self.ffn     = FNN(__C)
        
        self.dropout_1 = torch.nn.Dropout(__C.DROPOUT_R)
        self.norm_1 = LayerNorm(__C.HIDDEN_SIZE)
        
        self.dropout_2 = torch.nn.Dropout(__C.DROPOUT_R)
        self.norm_2 = LayerNorm(__C.HIDDEN_SIZE)
        
        
    def forward(self, x, y, x_mask, y_mask):
        x = self.norm_1(x + self.dropout_1(self.mhatt_2(value = y, key = y, query = x, mask = y_mask)))
        x = self.norm_2(x + self.dropout_2(self.ffn(x)))
        return x